import numpy as np
from petsc4py import PETSc
import os


class PetscMesh:
    def __init__(self, filepath):
        """
        Initializes the PetscMesh with a path to a .msh file.
        The DM is stored to ensure ordering is preserved during I/O.
        """
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Mesh file not found: {filepath}")

        # Load the DM once to establish the canonical PETSc ordering
        self.dm = PETSc.DMPlex().createFromFile(self.filepath, comm=PETSc.COMM_WORLD)
        self.dm.setUp()

    @classmethod
    def from_gmsh(cls, filepath):
        """Factory method to create a PetscMesh instance."""
        return cls(filepath)

    import os


    def to_h5(self, output_path, model=None):
        # 1. Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. Create the HDF5 Viewer
        viewer = PETSc.Viewer().createHDF5(
            output_path, mode=PETSc.Viewer.Mode.WRITE, comm=PETSc.COMM_WORLD
        )

        try:
            # Save the topology
            self.dm.view(viewer)

            if model is not None:
                # Prepare and evaluate ICs
                (cStart, cEnd) = self.dm.getHeightStratum(0)
                n_cells = cEnd - cStart

                # evaluate coordinates
                X_coords = np.zeros((self.dm.getDimension(), n_cells))
                for i, c in enumerate(range(cStart, cEnd)):
                    _, center, _ = self.dm.computeCellGeometryFVM(c)
                    X_coords[:, i] = center[: self.dm.getDimension()]

                # Handle Solution (Q)
                vec_q = self.dm.createGlobalVector()
                vec_q.setName("Solution")
                q_init = np.zeros((model.n_dof_q, n_cells))
                model.initial_conditions.apply(X_coords, q_init)
                vec_q.setArray(q_init.T.flatten())

                # View the vector
                viewer.view(vec_q)

                # Handle Auxiliary (Qaux)
                if hasattr(model, "n_dof_qaux") and model.n_dof_qaux > 0:
                    vec_aux = self.dm.createGlobalVector()
                    vec_aux.setName("Auxiliary")
                    qaux_init = np.ones((model.n_dof_qaux, n_cells))  # Default 1.0 like C++
                    vec_aux.setArray(qaux_init.T.flatten())
                    viewer.view(vec_aux)

            # 3. Explicitly flush and close before the object goes out of scope
            viewer.flush()

        finally:
            # Using finally ensures the file handle is closed even if an error occurs mid-write
            viewer.destroy()
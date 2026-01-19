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
    
    
    def to_h5(self, output_path, model=None):
        import h5py
        import numpy as np
        import os

        if model is None:
            return

        print(f"--- Exporting IC: {output_path} ---")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Compute Data (Serial Order)
        (cStart, cEnd) = self.dm.getHeightStratum(0)
        n_cells = cEnd - cStart
        dim = self.dm.getDimension()

        X_coords = np.zeros((dim, n_cells))
        for i, c in enumerate(range(cStart, cEnd)):
            _, center, _ = self.dm.computeCellGeometryFVM(c)
            X_coords[:, i] = center[:dim]

        q_init = np.zeros((model.n_variables, n_cells))
        model.initial_conditions.apply(X_coords, q_init)
        data_flat = q_init.T.flatten()

        # 2. Write h5py
        with h5py.File(output_path, "w") as f:
            f.attrs["Time"] = 0.0
            dset = f.create_dataset("Solution", data=data_flat)
            # Use b"seq" for NumPy 2.0 compatibility
            dset.attrs["vector_type"] = b"seq"

            if model.n_aux_variables > 0:
                aux_data = np.ones(n_cells * model.n_aux_variables)
                dset_aux = f.create_dataset("Auxiliary", data=aux_data)
                dset_aux.attrs["vector_type"] = b"seq"

        print("--- Python Export Complete ---")
        
    def to_h5_cloud(self, output_path, model=None):
        """
        Writes a coordinate-based point cloud HDF5 file.
        Robust against index reordering.
        """
        import h5py
        import numpy as np
        import os

        if model is None: return
        print(f"--- Exporting IC Cloud: {output_path} ---")
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Compute Centers & Data (Serial)
        (cStart, cEnd) = self.dm.getHeightStratum(0)
        n_cells = cEnd - cStart
        dim = self.dm.getDimension()

        # Arrays for Cloud
        centers = np.zeros((n_cells, 3)) # Always 3D for consistency
        for i, c in enumerate(range(cStart, cEnd)):
            _, center, _ = self.dm.computeCellGeometryFVM(c)
            centers[i, :dim] = center[:dim]

        q_init = np.zeros((model.n_variables, n_cells))
        model.initial_conditions.apply(centers[:, :dim].T, q_init)
        
        # 2. Write Simple H5 (No PETSc metadata needed)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset("centers", data=centers) # (N, 3)
            f.create_dataset("values", data=q_init.T) # (N, Vars)
            
            if model.n_aux_variables > 0:
                aux = np.ones((n_cells, model.n_aux_variables))
                f.create_dataset("aux", data=aux)

        print("--- Cloud Export Complete ---")
import unittest
import numpy as np
import pandas as pd

class TestReturnMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_filename = 'ret_matrix.csv'
        cls.ret_df = pd.read_csv(cls.csv_filename, index_col=0, parse_dates=True)
        cls.p, cls.n = cls.ret_df.shape

    def test_mean_centering_in_covariance(self):
        """
        check if cov centers are mean'd zero
        """
        subset_size = 50
        subset = self.ret_df.iloc[:subset_size].values

        # Method 1: pandas cov()
        cov_pandas = pd.DataFrame(subset).cov().values

        # Method 2: Manual mean-centering and computation
        centered_data = subset - subset.mean(axis=1)[:, np.newaxis]
        cov_manual = (centered_data @ centered_data.T) / (self.n - 1)

        np.testing.assert_allclose(cov_pandas, cov_manual, atol=1e-8,
                                 err_msg="Covariance matrices don't match after centering")

        # Print first few means to verify centering
        print("\nMean check for centered data:")
        print(f"Max absolute mean in centered data: {np.abs(centered_data.mean(axis=1)).max():.2e}")

    def test_YtY_identity(self):
        """
        Test (1/n) Y^T Y = (1/n) sum(r_i r_i^T)
        """
        subset_size = 10
        subset = self.ret_df.iloc[:subset_size].values
        n = self.n

        # Method 1: Direct Y^T Y
        YtY_direct = subset.T @ subset / n

        # Method 2: Sum of outer products
        YtY_sum = np.zeros((n, n))
        for i in range(subset_size):
            r_i = subset[i]
            YtY_sum += np.outer(r_i, r_i)
        YtY_sum /= n

        np.testing.assert_allclose(YtY_direct, YtY_sum, atol=1e-8,
                                 err_msg="Y^T Y identity failed")

    def test_max_eigenvalue_identity(self):
        """
        Test lambda_max^2 = max_u (1/n) sum((r_i^T u)^2) subject to |u| = 1
        """
        subset_size = 100
        subset = self.ret_df.iloc[:subset_size].values
        n = self.n

        # Compute Y^T Y / n
        YtY = subset.T @ subset / n

        # Get largest eigenvalue and corresponding eigenvector
        eigenvals, eigenvecs = np.linalg.eigh(YtY)
        lambda_max = np.sqrt(eigenvals[-1])
        u = eigenvecs[:, -1]

        # Verify u is unit vector
        self.assertAlmostEqual(np.linalg.norm(u), 1.0, delta=1e-8)

        # Compute sum((r_i^T u)^2) / n
        projections = subset @ u
        sum_squared_projections = np.sum(projections ** 2) / n

        # lambda_max^2 should equal sum of squared projections
        np.testing.assert_allclose(lambda_max ** 2, sum_squared_projections, atol=1e-8,
                                 err_msg="Max eigenvalue identity failed")

    def test_eigenvalue_increase_property(self):
        """
        Test lambda_max_new^2 >= lambda_max^2 + (1/n)(r_new^T u)^2
        """
        num_tests = 100
        p_initial = 100

        for test_case in range(num_tests):
            np.random.seed(test_case)
            initial_indices = np.random.choice(self.p, p_initial, replace=False)
            Y = self.ret_df.iloc[initial_indices].values

            # Get a new row not in initial set
            remaining_indices = list(set(range(self.p)) - set(initial_indices))
            next_index = np.random.choice(remaining_indices)
            r_new = self.ret_df.iloc[next_index].values

            # Current maximum eigenvalue and eigenvector
            YtY = Y.T @ Y / self.n
            eigenvals, eigenvecs = np.linalg.eigh(YtY)
            lambda_max_squared = eigenvals[-1]
            u = eigenvecs[:, -1]

            # Compute (1/n)(r_new^T u)^2
            projection_squared = (r_new @ u) ** 2 / self.n

            # Add new row and compute new maximum eigenvalue
            Y_extended = np.vstack([Y, r_new])
            YtY_new = Y_extended.T @ Y_extended / self.n
            lambda_max_new_squared = np.max(np.real(np.linalg.eigvals(YtY_new)))

            # Verify inequality
            self.assertGreaterEqual(
                lambda_max_new_squared,
                lambda_max_squared + projection_squared - 1e-8,
                msg=f"Test case {test_case}: Eigenvalue increase property failed"
            )

            if test_case == 0:
                print("\nFirst test case details:")
                print(f"Initial lambda_max^2: {lambda_max_squared:.10f}")
                print(f"(1/n)(r_new^T u)^2: {projection_squared:.10f}")
                print(f"New lambda_max^2: {lambda_max_new_squared:.10f}")
                print(f"Difference from minimum: "
                      f"{lambda_max_new_squared - (lambda_max_squared + projection_squared):.10f}")

    def test_positive_semidefinite(self):
        """
        Test that Y^T Y / n is positive semidefinite
        """
        subset_size = 200
        subset = self.ret_df.iloc[:subset_size].values

        YtY = subset.T @ subset / self.n
        eigenvalues = np.real(np.linalg.eigvals(YtY))
        
        self.assertTrue(np.all(eigenvalues >= -1e-8),
                       msg="Y^T Y / n has negative eigenvalues")

if __name__ == '__main__':
    unittest.main()
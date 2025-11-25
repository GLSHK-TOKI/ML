import unittest

# Discover and load all the test files in the tests folder
loader = unittest.TestLoader()
suite = loader.discover(start_dir="tests", pattern="test_*.py")

# Run the test suite
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
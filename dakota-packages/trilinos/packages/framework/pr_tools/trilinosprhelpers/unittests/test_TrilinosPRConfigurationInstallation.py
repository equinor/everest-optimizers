#!/usr/bin/env python3
""" """



try:  # pragma: no cover
    import builtins  # pragma: no cover
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

import copy
import os
import sys

if sys.version_info >= (3, 0):  # pragma: no cover
    pass  # pragma: no cover
else:  # pragma: no cover
    pass  # pragma: no cover

import unittest
from unittest import TestCase

try:  # pragma: no cover
    from unittest.mock import patch
except:  # pragma: no cover
    from unittest.mock import patch

import argparse

# import subprocess
# from trilinosprhelpers import setenvironment
# from trilinosprhelpers import sysinfo
import trilinosprhelpers


# ==============================================================================
#
#                         M O C K   H E L P E R S
#
# ==============================================================================
def mock_chdir(*args, **kwargs):
    print(f"MOCK> os.chdir('{args[0]}')")
    return 0


def mock_subprocess_check_call(*args, **kwargs):
    cmd = " ".join(args[0])
    print(f"MOCK> subprocess.check_call({cmd})")
    return 0


def mock_subprocess_check_output(*args, **kwargs):
    """Mock out a subprocess.check_output()
    """
    params = copy.deepcopy(args[0])
    for k, v in kwargs.items():
        params.append(f"{k}={v}")
    output = "--- subprocess.check_output({})".format(", ".join(params))

    print("MOCK> mock_packageEnables_check_output()")
    for k in args[0]:  # pragma: no cover
        print(f"    - '{k}'")  # pragma: no cover
    for k, v in kwargs.items():  # pragma: no cover
        print(f"    - {k}={v}")  # pragma: no cover
    print()
    return str.encode(output)


def mock_module_apply(*args, **kwargs):
    """Simple function call mock that returns 0
    """
    cmd = ", ".join([f"'{x}'" for x in args])  # pragma: no cover
    print(f"MOCK> module({cmd})")  # pragma: no cover
    return 0  # pragma: no cover


# ==============================================================================
#
#                                T E S T S
#
# ==============================================================================
class TrilinosPRConfigurationInstallationTest(TestCase):
    """Test TrilinsoPRConfigurationInstallation class
    """

    def setUp(self):
        os.environ["PULLREQUEST_CDASH_TRACK"] = "Pull Request"

        # Find the config file
        env_config_file = "trilinos_pr_test.ini"
        self._env_config_file = self.find_config_ini(env_config_file)
        gen_config_file = "gen-config.ini"
        self._gen_config_file = self.find_config_ini(gen_config_file)

        # Set up dummy command line arguments
        self._args = self.dummy_args()

        # Set up some global mock patches
        self.patch_cpu_count = patch("multiprocessing.cpu_count", return_value=64)
        self.mock_cpu_count = self.patch_cpu_count.start()

        self.patch_os_chdir = patch("os.chdir", side_effect=mock_chdir)
        self.mock_chdir = self.patch_os_chdir.start()

        self.patch_subprocess_check_call = patch(
            "subprocess.check_call", side_effect=mock_subprocess_check_call,
        )
        self.mock_subprocess_check_call = self.patch_subprocess_check_call.start()

        self.patch_subprocess_check_output = patch(
            "subprocess.check_output", side_effect=mock_subprocess_check_output,
        )
        self.mock_subprocess_check_output = self.patch_subprocess_check_output.start()

        self.patch_modulehelper_module = patch(
            "setenvironment.ModuleHelper.module", side_effect=mock_module_apply,
        )
        self.mock_modulehelper_module = self.patch_modulehelper_module.start()

    def tearDown(self):
        del os.environ["PULLREQUEST_CDASH_TRACK"]

        # Shut down global mock patches
        self.patch_cpu_count.stop()
        self.patch_os_chdir.stop()
        self.patch_subprocess_check_call.stop()
        self.patch_subprocess_check_output.stop()
        self.patch_modulehelper_module.stop()

    def dummy_args(self):
        """Generate dummy command line arguments
        """
        output = argparse.Namespace(
            source_repo_url="https://github.com/trilinos/Trilinos",
            target_repo_url="https://github.com/trilinos/Trilinos",
            target_branch_name="develop",
            pullrequest_build_name="Trilinos-pullrequest-gcc-installation-testing",
            genconfig_build_name="rhel8_sems-gnu-openmpi_release_static_no-kokkos-arch_no-asan_no-complex_no-fpic_mpi_no-pt_no-rdc_no-package-enables",
            dashboard_build_name="gnu-openmpi_release_static",
            pullrequest_cdash_track="Pull Request",
            jenkins_job_number=99,
            pullrequest_number="0000",
            pullrequest_env_config_file=self._env_config_file,
            pullrequest_gen_config_file=self._gen_config_file,
            workspace_dir=".",
            build_dir="build",
            source_dir="source",
            ctest_driver="ctest_driver.cmake",
            ctest_drop_site="testint.sandia.gov",
            filename_packageenables="../packageEnables.cmake",
            filename_subprojects="../package_subproject_list.cmake",
            skip_create_packageenables=False,
            mode="standard",
            req_mem_per_core=3.0,
            max_cores_allowed=12,
            num_concurrent_tests=-1,
            ccache_enable=False,
            dry_run=False,
            use_explicit_cachefile=False,
            extra_configure_args="",
        )
        return output

    #    def dummy_args_python2(self):
    #        """
    #        Extend dummy args to change the pullrequest_build_name to use the
    #        Python 2.x test set.
    #        """
    #        args = copy.deepcopy(self.dummy_args())                         # pragma: no cover
    #        args.pullrequest_build_name = "Trilinos_pullrequest_python_2"   # pragma: no cover
    #        return args                                                     # pragma: no cover

    def dummy_args_dry_run(self):
        """Extend dummy args to enable dry-run mode.
        """
        args = copy.deepcopy(self.dummy_args())  # pragma: no cover
        args.dry_run = True  # pragma: no cover
        return args  # pragma: no cover

    def find_config_ini(self, filename="config.ini"):
        rootpath = "."
        output = None
        for dirpath, dirnames, filename_list in os.walk(rootpath):
            if filename in filename_list:
                output = os.path.join(dirpath, filename)
                break
        return output

    def test_TrilinosPRConfigurationInstallationExec(self):
        """Test the Installation Configuration
        """
        print()
        args = self.dummy_args()
        pr_config = trilinosprhelpers.TrilinosPRConfigurationInstallation(args)

        # prepare step
        ret = pr_config.prepare_test()
        self.assertEqual(ret, 0)
        self.mock_cpu_count.assert_called()

        # execute step
        with self.assertRaises(NotImplementedError):
            ret = pr_config.execute_test()
        # self.mock_chdir.assert_called_once()
        # self.mock_subprocess_check_call.assert_called_once()
        # self.assertEqual(ret, 0)
        # self.assertTrue(Path(os.path.join(args.workspace_dir, "generatedPRFragment.cmake")).is_file())
        # os.unlink(os.path.join(args.workspace_dir, "generatedPRFragment.cmake"))

        # self.assertTrue(Path(os.path.join(args.workspace_dir, "packageEnables.cmake")).is_file())
        # os.unlink(os.path.join(args.workspace_dir, "packageEnables.cmake"))

        # self.assertTrue(Path(os.path.join(args.workspace_dir, "package_subproject_list.cmake")).is_file())
        # os.unlink(os.path.join(args.workspace_dir, "package_subproject_list.cmake"))

    def test_TrilinosPRConfigurationInstallationDryRun(self):
        """Test the Installation Configuration
        """
        args = self.dummy_args_dry_run()
        pr_config = trilinosprhelpers.TrilinosPRConfigurationInstallation(args)

        # prepare step
        ret = pr_config.prepare_test()
        self.assertEqual(ret, 0)
        self.mock_cpu_count.assert_called()

        # execute step
        ret = pr_config.execute_test()
        self.assertEqual(ret, 0)


if __name__ == "__main__":
    unittest.main()  # pragma nocover

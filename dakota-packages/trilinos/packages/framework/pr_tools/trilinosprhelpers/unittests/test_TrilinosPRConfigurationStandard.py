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
from pathlib import Path

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
    if not isinstance(params, list):
        params = [params]
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
    """Mock handler for ModuleHelper.module() calls.
    """
    cmd = ", ".join([f"'{x}'" for x in args])  # pragma: no cover
    print(f"MOCK> module({cmd})")  # pragma: no cover
    return 0


# ==============================================================================
#
#                                T E S T S
#
# ==============================================================================
class TrilinosPRConfigurationStandardTest(TestCase):
    """Test TrilinsoPRConfigurationStandard class
    """

    def setUp(self):
        os.environ["PULLREQUEST_CDASH_TRACK"] = "Pull Request"

        # Find the config files
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
            pullrequest_build_name="Trilinos-pullrequest-gcc",
            genconfig_build_name="rhel8_sems-gnu-openmpi_release_static_no-kokkos-arch_no-asan_no-complex_no-fpic_mpi_no-pt_no-rdc_no-package-enables",
            dashboard_build_name="gnu-openmpi_release_static",
            pullrequest_cdash_track="Pull Request",
            jenkins_job_number=99,
            pullrequest_number="0000",
            pullrequest_env_config_file=self._env_config_file,
            pullrequest_gen_config_file=self._gen_config_file,
            workspace_dir=".",
            source_dir="source",
            build_dir="build",
            ctest_driver="ctest_driver.cmake",
            ctest_drop_site="testing.sandia.gov",
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
            skip_run_tests=False,
        )
        return output

    def find_config_ini(self, filename="trilinos_pr_test.ini"):
        rootpath = "."
        output = None
        for dirpath, dirnames, filename_list in os.walk(rootpath):
            if filename in filename_list:
                output = os.path.join(dirpath, filename)
                break
        return output

    def test_TrilinosPRConfigurationStandardExec(self):
        """Test the Standard Configuration
        """
        print()
        args = self.dummy_args()
        pr_config = trilinosprhelpers.TrilinosPRConfigurationStandard(args)

        # prepare step
        ret = pr_config.prepare_test()
        self.assertEqual(ret, 0)
        self.mock_cpu_count.assert_called()

        # execute step
        ret = pr_config.execute_test()
        self.mock_chdir.assert_called_once()
        self.mock_subprocess_check_call.assert_called()
        self.assertEqual(ret, 0)
        self.assertTrue(
            Path(
                os.path.join(args.workspace_dir, "generatedPRFragment.cmake"),
            ).is_file(),
        )
        os.unlink(os.path.join(args.workspace_dir, "generatedPRFragment.cmake"))

    def test_TrilinosPRConfigurationStandardDryRun(self):
        """Test the Standard Configuration
        - Change args to enable dry_run mode.
        """
        args = self.dummy_args()
        args.dry_run = True
        pr_config = trilinosprhelpers.TrilinosPRConfigurationStandard(args)

        # prepare step
        ret = pr_config.prepare_test()
        self.assertEqual(ret, 0)
        self.mock_cpu_count.assert_called()

        # execute step
        ret = pr_config.execute_test()
        self.assertEqual(ret, 0)

    def test_TrilinosPRConfigurationStandardPython3(self):
        """Test the Standard Configuration
        - Change args to enable:
            - pullrequest_build_name = "Trilinos-pullrequest-python_3"
            - dry_run = True
        - Change args to enable dry_run mode.
        """
        args = self.dummy_args()
        args.pullrequest_build_name = "Trilinos_PR_python3"
        pr_config = trilinosprhelpers.TrilinosPRConfigurationStandard(args)

        # prepare step
        ret = pr_config.prepare_test()
        self.assertEqual(ret, 0)
        self.mock_cpu_count.assert_called()
        self.assertTrue(
            Path(os.path.join(args.workspace_dir, "packageEnables.cmake")).is_file(),
        )
        os.unlink(os.path.join(args.workspace_dir, "packageEnables.cmake"))
        self.assertTrue(
            Path(
                os.path.join(args.workspace_dir, "package_subproject_list.cmake"),
            ).is_file(),
        )
        os.unlink(os.path.join(args.workspace_dir, "package_subproject_list.cmake"))

        # execute step
        # ret = pr_config.execute_test()
        # self.assertEqual(ret, 0)


if __name__ == "__main__":
    unittest.main()  # pragma nocover

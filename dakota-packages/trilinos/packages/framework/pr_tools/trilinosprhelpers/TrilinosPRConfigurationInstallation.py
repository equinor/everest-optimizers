#!/usr/bin/env python3
# -*- mode: python; py-indent-offset: 4; py-continuation-offset: 4 -*-
"""Custom PR Executor for Installation testing
"""

import os
import subprocess
from pathlib import Path

from gen_config import GenConfig

from . import TrilinosPRConfigurationBase


class TrilinosPRConfigurationInstallation(TrilinosPRConfigurationBase):
    """Implements Standard mode Trilinos Pull Request Driver
    """

    def __init__(self, args):
        super(TrilinosPRConfigurationInstallation, self).__init__(args)

    def execute_test(self):
        """Execute the test
        """
        print("+" + "=" * 78 + "+")
        print(
            "|   E X E C U T E   I N S T A L L A T I O N   P U L L R E Q U E S T   T E S T",
        )
        print("+" + "=" * 78 + "+")

        #
        # Typically, we execute the test from $WORKSPACE/TFW_testing_single_configure_prototype
        # We'll skip it if we're doing a dry-run.
        #
        print()
        print(f"--- Change directory to {self.working_directory_ctest}")
        if not self.args.dry_run:
            os.chdir(self.working_directory_ctest)
        print("--- OK")
        print()

        # Use GenConfig to write the configure script for cmake
        genconfig_arglist = [
            "-y",
            "--force",
            "--cmake-fragment",
            os.path.join(self.arg_workspace_dir, self.config_script),
            self.arg_pr_genconfig_job_name,
        ]
        genconfig_inifile = Path(self.arg_pr_gen_config_file)

        gc = GenConfig(genconfig_arglist, gen_config_ini_file=(genconfig_inifile))
        # TODO: The tuple around `genconfig_inifile` should not be needed b/c ("A") == "A"
        #       since a tuple of size 1 is generally just the object. To actually get a single
        #       entry tuple, you'd need to do this ``tuple("A")`` which would result in ``('A',)``

        if not self.args.dry_run:
            gc.write_cmake_fragment()

        # Execute the call to ctest.
        # - NOTE: simple_testing.cmake can be found in the TFW_single_configure_support_scripts
        #         repository.
        cmd = [
            "ctest",
            "-S",
            "simple_testing.cmake",
            f"-Dbuild_name={self.pullrequest_build_name}",
            "-Dskip_by_parts_submit=OFF",
            "-Dskip_update_step=ON",
            f"-Ddashboard_model={self.dashboard_model}",
            f"-Ddashboard_track={self.arg_pullrequest_cdash_track}",
            f"-DPARALLEL_LEVEL={self.concurrency_build}",
            f"-DTEST_PARALLEL_LEVEL={self.concurrency_test}",
            f"-Dbuild_dir={self.arg_workspace_dir}/pull_request_test",
            "-Dconfigure_script="
            + os.path.join(self.arg_workspace_dir, self.config_script),
            "-Dpackage_enables=" + self.arg_filename_packageenables,
            "-Dsubprojects_file=" + self.arg_filename_subprojects,
        ]

        print("--- ctest command:")
        print("--- cmd = {}".format(" \\\n   ".join(cmd)))
        print("--- ")

        if not self.args.dry_run:
            raise NotImplementedError("This is just stub code -- do not execute")
            try:
                subprocess.check_call(cmd, env=os.environ)
                # Note: check_call will throw an exception if there's a problem.
            except:
                print("--- ctest command failed!")
                return 1
        print("--- SKIPPED DUE TO DRYRUN")
        print()

        return 0

install_test_dependencies () {
    pip install pytest
}

start_tests () {
    set -e
    pytest

    # Integration test with everest
    git clone --depth 1 https://github.com/equinor/ert __ert
    cp -a __ert/test-data/everest/math_func .
    rm -rf __ert

    pushd math_func
    perl -p -i -e 's/optpp_q_newton/bcq_newton/' config_minimal.yml
    everest run config_minimal.yml
    popd
}

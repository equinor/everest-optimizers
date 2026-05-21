install_test_dependencies () {
    pip install pytest
}

start_everest_integration_tests () {
    rm -rf math_func
    git clone --depth 1 https://github.com/equinor/ert __ert
    cp -a __ert/test-data/everest/math_func .
    rm -rf __ert

    pushd math_func

    sed -i  's/optpp_q_newton/bcq_newton/' config_minimal.yml
    sed -i '/max_batch_num/d' config_minimal.yml
    everest run config_minimal.yml

    sed -i  's/optpp_q_newton/q_nips/' config_advanced.yml
    sed -i '/max_batch_num/d' config_advanced.yml
    everest run config_advanced.yml

    popd
}

start_tests () {
    set -e
    pytest
    start_everest_integration_tests
}

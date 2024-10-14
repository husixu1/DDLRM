#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <map>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "utils/log.h"
#include "cuda_metrics/measureMetricPW.hpp"


// return values
enum dp_retval_t {
    DP_SUCCESS = 0,
    DP_FAILED
};

static constexpr int kDprofile_Metric_DramRead = 0;
static constexpr int kDprofile_Metric_DramWrite = 1;
static constexpr int kDprofile_Metric_L1Load = 2;
static constexpr int kDprofile_Metric_L1Store = 3;
static constexpr int kDprofile_Metric_L2Load = 4;
static constexpr int kDprofile_Metric_L2Store = 5;


class DProfile {
 public:
    DProfile(){}
    ~DProfile() = default;

    /*!
     *  \brief  start profile the specified metric
     *  \param  metric_types set of metrics to be profiled
     */
    void start_profile(std::set<int> metric_set);
    
    /*!
     *  \brief  stop profile and collect message
     *  \return map of collected metrics
     */
    std::map<std::string, double> stop_profile();

 private:
    
    std::set<int> _issued_metric_set;
    std::vector<std::string> _issued_cupti_metrics;
};

PYBIND11_MODULE(libdp, m) {
    pybind11::class_<DProfile>(m, "DProfile")
        .def(pybind11::init<>())
        .def("start_profile", &DProfile::start_profile)
        .def("stop_profile", &DProfile::stop_profile);
}

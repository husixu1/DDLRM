#include "dprofile.hpp"


void DProfile::start_profile(std::set<int> metric_set){
    typename std::set<int>::iterator set_iter;
    this->_issued_metric_set = metric_set;
    this->_issued_cupti_metrics.clear();
    
    for(set_iter = this->_issued_metric_set.begin(); set_iter != this->_issued_metric_set.end(); set_iter++){
        switch (*set_iter)
        {
        case kDprofile_Metric_DramRead:
            this->_issued_cupti_metrics.push_back("dram__sectors_read.sum");
            this->_issued_cupti_metrics.push_back("dram__bytes_read.sum");
            this->_issued_cupti_metrics.push_back("dram__bytes_read.sum.per_second");
            break;
        case kDprofile_Metric_DramWrite:
            this->_issued_cupti_metrics.push_back("dram__sectors_write.sum");
            this->_issued_cupti_metrics.push_back("dram__bytes_write.sum");
            this->_issued_cupti_metrics.push_back("dram__bytes_write.sum.per_second");
            break;
        case kDprofile_Metric_L1Load:
            this->_issued_cupti_metrics.push_back("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum");
            this->_issued_cupti_metrics.push_back("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum");
            this->_issued_cupti_metrics.push_back("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second");
            this->_issued_cupti_metrics.push_back("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio");
            break;
        case kDprofile_Metric_L1Store:
            this->_issued_cupti_metrics.push_back("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum");
            this->_issued_cupti_metrics.push_back("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum");
            this->_issued_cupti_metrics.push_back("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second");
            this->_issued_cupti_metrics.push_back("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio");
            break;
        case kDprofile_Metric_L2Load:
            this->_issued_cupti_metrics.push_back("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum");
            this->_issued_cupti_metrics.push_back("lts__t_sectors_op_read.sum");
            this->_issued_cupti_metrics.push_back("lts__t_sectors_op_read.sum.per_second");
            break;
        case kDprofile_Metric_L2Store:
            this->_issued_cupti_metrics.push_back("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum");
            this->_issued_cupti_metrics.push_back("lts__t_sectors_op_write.sum");
            this->_issued_cupti_metrics.push_back("lts__t_sectors_op_write.sum.per_second");
            break;
        default:
            DPROFILE_ERROR_C("not supported metric type: %u", *set_iter)
        }
    }

    measureMetricsStart(this->_issued_cupti_metrics);
}


std::map<std::string, double> DProfile::stop_profile(){
    typename std::set<int>::iterator set_iter;
    std::vector<double> metrics;
    std::map<std::string, double> ret_map;

    uint64_t list_ptr = 0;

    metrics = measureMetricsStop();

    for(set_iter = this->_issued_metric_set.begin(); set_iter != this->_issued_metric_set.end(); set_iter++){
        switch (*set_iter)
        {
        case kDprofile_Metric_DramRead:
            ret_map["s_dram_read_transactions"] = metrics[list_ptr];
            ret_map["d_dram_read_transactions"] = metrics[list_ptr+1];
            ret_map["s_dram_read_bytes"] = metrics[list_ptr + 2];
            ret_map["d_dram_read_bytes"] = metrics[list_ptr + 3];
            ret_map["s_dram_read_throughput"] = metrics[list_ptr + 4];
            ret_map["d_dram_read_throughput"] = metrics[list_ptr + 5];
            list_ptr += 6;
            break;
        case kDprofile_Metric_DramWrite:
            ret_map["s_dram_write_transactions"] = metrics[list_ptr];
            ret_map["d_dram_write_transactions"] = metrics[list_ptr + 1];
            ret_map["s_dram_write_bytes"] = metrics[list_ptr + 2];
            ret_map["d_dram_write_bytes"] = metrics[list_ptr + 3];
            ret_map["s_dram_write_throughput"] = metrics[list_ptr + 4];
            ret_map["d_dram_write_throughput"] = metrics[list_ptr + 5];
            list_ptr += 6;
            break;
        case kDprofile_Metric_L1Load:
            ret_map["s_gld_transactions"] = metrics[list_ptr];
            ret_map["d_gld_transactions"] = metrics[list_ptr + 1];
            ret_map["s_global_load_requests"] = metrics[list_ptr + 2];
            ret_map["d_global_load_requests"] = metrics[list_ptr + 3];
            ret_map["s_gld_throughput"] = metrics[list_ptr + 4];
            ret_map["d_gld_throughput"] = metrics[list_ptr + 5];
            ret_map["s_gld_transactions_per_request"] = metrics[list_ptr + 6];
            ret_map["d_gld_transactions_per_request"] = metrics[list_ptr + 7];
            list_ptr += 8;
            break;
        case kDprofile_Metric_L1Store:
            ret_map["s_gst_transactions"] = metrics[list_ptr];
            ret_map["d_gst_transactions"] = metrics[list_ptr + 1];
            ret_map["s_global_store_requests"] = metrics[list_ptr + 2];
            ret_map["d_global_store_requests"] = metrics[list_ptr + 3];
            ret_map["s_gst_throughput"] = metrics[list_ptr + 4];
            ret_map["d_gst_throughput"] = metrics[list_ptr + 5];
            ret_map["s_gst_transactions_per_request"] = metrics[list_ptr + 6];
            ret_map["d_gst_transactions_per_request"] = metrics[list_ptr + 7];
            list_ptr += 8;
            break;
        case kDprofile_Metric_L2Load:
            ret_map["s_l2_global_load_bytes"] = metrics[list_ptr];
            ret_map["d_l2_global_load_bytes"] = metrics[list_ptr + 1];
            ret_map["s_l2_read_transactions"] = metrics[list_ptr + 2];
            ret_map["d_l2_read_transactions"] = metrics[list_ptr + 3];
            ret_map["s_l2_read_throughput"] = metrics[list_ptr + 4];
            ret_map["d_l2_read_throughput"] = metrics[list_ptr + 5];
            list_ptr += 6;
            break;
        case kDprofile_Metric_L2Store:
            ret_map["s_l2_global_store_bytes"] = metrics[list_ptr];
            ret_map["d_l2_global_store_bytes"] = metrics[list_ptr + 1];
            ret_map["s_l2_write_transactions"] = metrics[list_ptr + 2];
            ret_map["d_l2_write_transactions"] = metrics[list_ptr + 3];
            ret_map["s_l2_write_throughput"] = metrics[list_ptr + 4];
            ret_map["d_l2_write_throughput"] = metrics[list_ptr + 5];
            list_ptr += 6;
            break;
        default:
            DPROFILE_ERROR_C("not supported metric type: %u", *set_iter)
        }
    }

    return ret_map;
}

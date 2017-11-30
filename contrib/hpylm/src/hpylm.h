#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <unordered_set>
#include <vector>
#include <cassert>
#include <fstream>
#include "common.h"
#include "sampler.h"
#include "node.h"

class HPYLM{
public:
    Node* _root;               // Route node of context tree
    int _depth;                // Maximum depth
    double _g0;                // Zero-gram Probability

    // Parameters related to the node of depth m
    vector<double> _d_m;        // Pitman-Yor Discount factor of process
    vector<double> _theta_m;    // Pitman-Yor Concentration of processes

    // "A Bayesian Interpretation of Interpolated Kneser-Ney" Appendix C - reference
    // http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf
    vector<double> _a_m;        // For estimating the parameter d of the beta distribution
    vector<double> _b_m;        // For estimating the parameter d of the beta distribution
    vector<double> _alpha_m;    // For estimating the parameter θ of the gamma distribution
    vector<double> _beta_m;     // For estimating the parameter θ of the gamma distribution

    HPYLM(int ngram = 2){
        // Note that the depth starts from 0
        // 2-gram if the maximum depth is 1. root(0) -> 2-gram(1)
        // 3-gram if the maximum depth is 2. root(0) -> 2-gram(1) -> 3-gram(2)
        _depth = ngram - 1;

        _root = new Node(0);
        _root->_depth = 0;    // Route has depth 0

        for(int n = 0;n < ngram;n++){
            _d_m.push_back(HPYLM_INITIAL_D);    
            _theta_m.push_back(HPYLM_INITIAL_THETA);
            _a_m.push_back(HPYLM_INITIAL_A);    
            _b_m.push_back(HPYLM_INITIAL_B);    
            _alpha_m.push_back(HPYLM_INITIAL_ALPHA);
            _beta_m.push_back(HPYLM_INITIAL_BETA);
        }
    }
    ~HPYLM(){
        _delete_node(_root);
    }
    void _delete_node(Node* node){
        for(auto &elem: node->_children){
            Node* child = elem.second;
            _delete_node(child);
        }
        delete node;
    }
    int ngram(){
        return _depth + 1;
    }
    void set_g0(double g0){
        _g0 = g0;
    }
    // Add index word word of word string to model
    bool add_customer_at_timestep(vector<id> &token_ids, int token_t_index){
        Node* node = find_node_by_tracing_back_context(token_ids, token_t_index, _depth, true);
        assert(node != NULL);
        assert(node->_depth == _depth);
        id token_t = token_ids[token_t_index];
        node->add_customer(token_t, _g0, _d_m, _theta_m);
        return true;
    }
    bool remove_customer_at_timestep(vector<id> &token_ids, int token_t_index){
        Node* node = find_node_by_tracing_back_context(token_ids, token_t_index, _depth, false);
        assert(node != NULL);
        assert(node->_depth == _depth);
        id token_t = token_ids[token_t_index];
        node->remove_customer(token_t);
        // Delete the node when there are no more customers
        if(node->need_to_remove_from_parent()){
            node->remove_from_parent();
        }
        return true;
    }
    // token - It traces backward from the position t of the column
    // token_ids:        [0, 1, 2, 3, 4, 5]
    // token_t_index:4          ^     ^
    // order_t: 2               |<- <-|
    Node* find_node_by_tracing_back_context(vector<id> &token_ids, int token_t_index, int order_t, bool generate_node_if_needed = false, bool return_middle_node = false){
        if(token_t_index - order_t < 0){
            return NULL;
        }
        Node* node = _root;
        for(int depth = 1;depth <= order_t;depth++){
            id context_token_id = token_ids[token_t_index - depth];
            Node* child = node->find_child_node(context_token_id, generate_node_if_needed);
            if(child == NULL){
                if(return_middle_node){
                    return node;
                }
                return NULL;
            }
            node = child;
        }
        return node;
    }
    double compute_Pw_h(vector<id> &token_ids, vector<id> context_token_ids){
        double p = 1;
        for(int n = 0;n < token_ids.size();n++){
            p *= compute_Pw_h(token_ids[n], context_token_ids);
            context_token_ids.push_back(token_ids[n]);
        }
        return p;
    }
    double compute_Pw_h(id token_id, vector<id> &context_token_ids){
        // HPYLM in fixed depth
        assert(context_token_ids.size() >= _depth);
        Node* node = find_node_by_tracing_back_context(context_token_ids, context_token_ids.size(), _depth, false, true);
        assert(node != NULL);
        return node->compute_Pw(token_id, _g0, _d_m, _theta_m);
    }
    double compute_Pw(id token_id){
        return _root->compute_Pw(token_id, _g0, _d_m, _theta_m);
    }
    double compute_Pw(vector<id> &token_ids){
        assert(token_ids.size() >= _depth + 1);
        double mult_pw_h = 1;
        vector<id> context_token_ids(token_ids.begin(), token_ids.begin() + _depth);
        for(int t = _depth;t < token_ids.size();t++){
            id token_id = token_ids[t];
            mult_pw_h *= compute_Pw_h(token_id, context_token_ids);;
            context_token_ids.push_back(token_id);
        }
        return mult_pw_h;
    }
    double compute_log_Pw(vector<id> &token_ids){
        if(token_ids.size() < _depth + 1) return 0.0;
        assert(token_ids.size() >= _depth + 1);
        double sum_pw_h = 0;
        vector<id> context_token_ids(token_ids.begin(), token_ids.begin() + _depth);
        for(int t = _depth;t < token_ids.size();t++){
            id token_id = token_ids[t];
            double pw_h = compute_Pw_h(token_id, context_token_ids);
            assert(pw_h > 0);
            sum_pw_h += log(pw_h);
            context_token_ids.push_back(token_id);
        }
        return sum_pw_h;
    }

    ///
    double compute_token_log_p(vector<id> &token_ids, vector<Node*> &nodes)
    {
        double sum_pw_h = 0;
        for(int i=0; i < token_ids.size(); i++)
        {
            double pw_h = 0.0;
            pw_h = compute_node_Pw_h(token_ids[i], nodes[i]);
            assert(pw_h > 0);
            sum_pw_h += log(pw_h);
        }
        return sum_pw_h;
    }

    double compute_node_Pw_h(id token_id, Node* &node){
        assert(node != NULL);
        return node->compute_Pw(token_id, _g0, _d_m, _theta_m);
    }
    ///

    double compute_log2_Pw(vector<id> &token_ids){
        assert(token_ids.size() >= _depth + 1);
        double sum_pw_h = 0;
        vector<id> context_token_ids(token_ids.begin(), token_ids.begin() + _depth);
        for(int t = _depth;t < token_ids.size();t++){
            id token_id = token_ids[t];
            double pw_h = compute_Pw_h(token_id, context_token_ids);
            assert(pw_h > 0);
            sum_pw_h += log2(pw_h);
            context_token_ids.push_back(token_id);
        }
        return sum_pw_h;
    }
    id sample_next_token(vector<id> &context_token_ids, unordered_set<id> &all_token_ids){
        Node* node = find_node_by_tracing_back_context(context_token_ids, context_token_ids.size(), _depth, false, true);
        assert(node != NULL);
        vector<id> token_ids;
        vector<double> pw_h_array;
        double sum = 0;
        for(id token_id: all_token_ids){
            if(token_id == ID_BOS){
                continue;
            }
            double pw_h = compute_Pw_h(token_id, context_token_ids);
            if(pw_h > 0){
                token_ids.push_back(token_id);
                pw_h_array.push_back(pw_h);
                sum += pw_h;
            }
        }
        if(token_ids.size() == 0){
            return ID_EOS;
        }
        if(sum == 0){
            return ID_EOS;
        }
        double normalizer = 1.0 / sum;
        double bernoulli = sampler::uniform(0, 1);
        double stack = 0;
        for(int i = 0;i < token_ids.size();i++){
            stack += pw_h_array[i] * normalizer;
            if(stack > bernoulli){
                return token_ids[i];
            }
        }
        return token_ids.back();
    }
    // "A Bayesian Interpretation of Interpolated Kneser-Ney" Appendix C
    // http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf
    void sum_auxiliary_variables_recursively(Node* node, vector<double> &sum_log_x_u_m, vector<double> &sum_y_ui_m, vector<double> &sum_1_y_ui_m, vector<double> &sum_1_z_uwkj_m){
        for(auto elem: node->_children){
            Node* child = elem.second;
            int depth = child->_depth;
            double d = _d_m[depth];
            double theta = _theta_m[depth];
            sum_log_x_u_m[depth] += child->auxiliary_log_x_u(theta);    // log(x_u)
            sum_y_ui_m[depth] += child->auxiliary_y_ui(d, theta);        // y_ui
            sum_1_y_ui_m[depth] += child->auxiliary_1_y_ui(d, theta);    // 1 - y_ui
            sum_1_z_uwkj_m[depth] += child->auxiliary_1_z_uwkj(d);        // 1 - z_uwkj

            sum_auxiliary_variables_recursively(child, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m);
        }
    }
    // Estimation of d and θ
    void sample_hyperparams(){
        int max_depth = _d_m.size() - 1;

        // Notice that the depth of the parent node is 0
        vector<double> sum_log_x_u_m(max_depth + 1, 0.0);
        vector<double> sum_y_ui_m(max_depth + 1, 0.0);
        vector<double> sum_1_y_ui_m(max_depth + 1, 0.0);
        vector<double> sum_1_z_uwkj_m(max_depth + 1, 0.0);

        // _root
        sum_log_x_u_m[0] = _root->auxiliary_log_x_u(_theta_m[0]);            // log(x_u)
        sum_y_ui_m[0] = _root->auxiliary_y_ui(_d_m[0], _theta_m[0]);        // y_ui
        sum_1_y_ui_m[0] = _root->auxiliary_1_y_ui(_d_m[0], _theta_m[0]);    // 1 - y_ui
        sum_1_z_uwkj_m[0] = _root->auxiliary_1_z_uwkj(_d_m[0]);                // 1 - z_uwkj

        // other than that
        sum_auxiliary_variables_recursively(_root, sum_log_x_u_m, sum_y_ui_m, sum_1_y_ui_m, sum_1_z_uwkj_m);

        for(int u = 0;u <= _depth;u++){
            _d_m[u] = sampler::beta(_a_m[u] + sum_1_y_ui_m[u], _b_m[u] + sum_1_z_uwkj_m[u]);
            _theta_m[u] = sampler::gamma(_alpha_m[u] + sum_y_ui_m[u], _beta_m[u] - sum_log_x_u_m[u]);
        }
    }
    int get_num_nodes(){
        return _root->get_num_nodes();
    }
    int get_num_customers(){
        return _root->get_num_customers();
    }
    int get_num_tables(){
        return _root->get_num_tables();
    }
    int get_sum_stop_counts(){
        return _root->sum_stop_counts();
    }
    int get_sum_pass_counts(){
        return _root->sum_pass_counts();
    }
    void count_tokens_of_each_depth(unordered_map<int, int> &map){
        _root->count_tokens_of_each_depth(map);
    }

    void enumerate_phrases_at_depth(int depth, vector<vector<id>> &phrases){
        assert(depth <= _depth);
        // Search nodes of specified depth
        vector<Node*> nodes;
        _root->enumerate_nodes_at_depth(depth, nodes);
        for(auto &node: nodes){
            vector<id> phrase;
            while(node->_parent){
                phrase.push_back(node->_token_id);
                node = node->_parent;
            }
            phrases.push_back(phrase);
        }
    }

    template <class Archive>
    void serialize(Archive& archive, unsigned int version)
    {
        archive & _root;
        archive & _depth;
        archive & _g0;
        archive & _d_m;
        archive & _theta_m;
        archive & _a_m;
        archive & _b_m;
        archive & _alpha_m;
        archive & _beta_m;
    }
    bool save(string filename = "hpylm.model"){
        std::ofstream ofs(filename);
        boost::archive::binary_oarchive oarchive(ofs);
        oarchive << *this;
        return true;
    }
    bool load(string filename = "hpylm.model"){
        std::ifstream ifs(filename);
        if(ifs.good() == false){
            return false;
        }
        boost::archive::binary_iarchive iarchive(ifs);
        iarchive >> *this;
        return true;
    }
};

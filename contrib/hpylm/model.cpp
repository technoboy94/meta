#include <boost/python.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <locale>
#include "src/node.h"
#include "src/hpylm.h"
#include "src/vocab.h"

using namespace boost;
using namespace std;

void split_word_by(const wstring &str, wchar_t delim, vector<wstring> &elems){
    elems.clear();
    wstring item;
    for(wchar_t ch: str){
        if (ch == delim){
            if (!item.empty()){
                elems.push_back(item);
            }
            item.clear();
        }
        else{
            item += ch;
        }
    }
    if (!item.empty()){
        elems.push_back(item);
    }
}


class PyHPYLM{
public:
    HPYLM* _hpylm;
    Vocab* _vocab;
    vector<vector<id>> _dataset_train;
    vector<vector<id>> _dataset_test;
    vector<int> _rand_indices;
    // statistics
    unordered_map<id, int> _word_count;
    int _sum_word_count;
    bool _gibbs_first_addition;
    PyHPYLM(){
        init(3);
    }
    PyHPYLM(int ngram){
        init(ngram);
    }
    ~PyHPYLM(){
        delete _hpylm;
        delete _vocab;
    }
    void init(int ngram){
        ios_base::sync_with_stdio(false);
        locale default_loc("en_US.UTF-8");
        locale::global(default_loc);
        locale ctype_default(locale::classic(), default_loc, locale::ctype);
        wcout.imbue(ctype_default);
        wcin.imbue(ctype_default);

        _hpylm = new HPYLM(++ngram);
        _vocab = new Vocab();
        _gibbs_first_addition = true;
        _sum_word_count = 0;
    }
    bool load_textfile(string filename, double train_split_ratio){
        wifstream ifs(filename.c_str());
        wstring sentence;
        if(ifs.fail()){
            return false;
        }
        vector<wstring> lines;
        while (getline(ifs, sentence) && !sentence.empty()){
            if (PyErr_CheckSignals() != 0) {        // check if ctrl+c  was pressed
                return false;
            }
            lines.push_back(sentence);
        }
        vector<int> rand_indices;
        for(int i = 0;i < lines.size();i++){
            rand_indices.push_back(i);
        }
        int train_split = lines.size() * train_split_ratio;
        shuffle(rand_indices.begin(), rand_indices.end(), sampler::mt);
        for(int i = 0;i < rand_indices.size();i++){
            wstring &sentence = lines[rand_indices[i]];
            if(i < train_split){
                add_train_data(sentence);
            }else{
                add_test_data(sentence);
            }
        }
        return true;
    }
    void add_train_data(wstring sentence){
        _add_data_to(sentence, _dataset_train);
    }
    void add_test_data(wstring sentence){
        _add_data_to(sentence, _dataset_test);
    }
    void _add_data_to(wstring &sentence, vector<vector<id>> &dataset){
        vector<wstring> word_str_array;
        split_word_by(sentence, L' ', word_str_array);
        if(word_str_array.size() > 0){
            vector<id> words;
            for(int i = 0;i < _hpylm->_depth;i++){
                words.push_back(ID_BOS);
            }
            for(auto word_str: word_str_array){
                if(word_str.size() == 0){
                    continue;
                }
                id token_id = _vocab->add_string(word_str);
                words.push_back(token_id);
                _word_count[token_id] += 1;
                _sum_word_count += 1;
            }
            words.push_back(ID_EOS);
            dataset.push_back(words);
        }
    }
    void set_g0(double g0){
        _hpylm->_g0 = g0;
    }
    void load(string dirname){
        _vocab->load(dirname + "/hpylm.vocab");
        if(_hpylm->load(dirname + "/hpylm.model")){
            _gibbs_first_addition = false;
        }
    }
    void save(string dirname){
        _vocab->save(dirname + "/hpylm.vocab");
        _hpylm->save(dirname + "/hpylm.model");
    }
    void perform_gibbs_sampling(){
        if(_rand_indices.size() != _dataset_train.size()){
            _rand_indices.clear();
            for(int data_index = 0;data_index < _dataset_train.size();data_index++){
                _rand_indices.push_back(data_index);
            }
        }
        shuffle(_rand_indices.begin(), _rand_indices.end(), sampler::mt);
        for(int n = 0;n < _dataset_train.size();n++){
            if (PyErr_CheckSignals() != 0) {
                return;
            }
            int data_index = _rand_indices[n];
            vector<id> &token_ids = _dataset_train[data_index];
            for(int token_t_index = _hpylm->ngram() - 1;token_t_index < token_ids.size();token_t_index++){
                if(_gibbs_first_addition == false){
                    _hpylm->remove_customer_at_timestep(token_ids, token_t_index);
                }
                _hpylm->add_customer_at_timestep(token_ids, token_t_index);
            }
        }
        _gibbs_first_addition = false;
    }
    void remove_all_data(){
        for(int data_index = 0;data_index < _dataset_train.size();data_index++){
            if (PyErr_CheckSignals() != 0) {
                return;
            }
            vector<id> &token_ids = _dataset_train[data_index];
            for(int token_t_index = _hpylm->ngram() - 1;token_t_index < token_ids.size();token_t_index++){
                _hpylm->remove_customer_at_timestep(token_ids, token_t_index);
            }
        }
    }
    int get_num_train_data(){
        return _dataset_train.size();
    }
    int get_num_test_data(){
        return _dataset_test.size();
    }
    int get_num_nodes(){
        return _hpylm->get_num_nodes();
    }
    int get_num_customers(){
        return _hpylm->get_num_customers();
    }
    int get_num_types_of_words(){
        return _word_count.size();
    }
    int get_num_words(){
        return _sum_word_count;
    }
    int get_hpylm_depth(){
        return _hpylm->_depth;
    }
    id get_bos_id(){
        return ID_BOS;
    }
    id get_eos_id(){
        return ID_EOS;
    }

    void sample_hyperparameters(){
        _hpylm->sample_hyperparams();
    }
    // Calculate the log likelihood of the entire data set
    double compute_log_Pdataset_train(){
        return _compute_log_Pdataset(_dataset_train);
    }
    double compute_log_Pdataset_test(){
        return _compute_log_Pdataset(_dataset_test);
    }
    double _compute_log_Pdataset(vector<vector<id>> &dataset){
        double log_Pdataset = 0;
        for(int data_index = 0;data_index < dataset.size();data_index++){
            if (PyErr_CheckSignals() != 0) {
                return 0;
            }
            vector<id> &token_ids = dataset[data_index];
            log_Pdataset += _hpylm->compute_log_Pw(token_ids);;
        }
        return log_Pdataset;
    }
    double compute_perplexity_train(){
        return _compute_perplexity(_dataset_train);
    }
    double compute_perplexity_test(){
        return _compute_perplexity(_dataset_test);
    }
    double _compute_perplexity(vector<vector<id>> &dataset){
        double log_Pdataset = 0;
        for(int data_index = 0;data_index < dataset.size();data_index++){
            if (PyErr_CheckSignals() != 0) {
                return 0;
            }
            vector<id> &token_ids = dataset[data_index];
            log_Pdataset += _hpylm->compute_log2_Pw(token_ids) / (token_ids.size() - _hpylm->_depth);
        }
        return pow(2.0, -log_Pdataset / (double)dataset.size());
    }
    wstring generate_sentence(){
        std::vector<id> context_token_ids;
        for(int i = 0;i < _hpylm->_depth;i++){
            context_token_ids.push_back(ID_BOS);
        }
        for(int n = 0;n < 1000;n++){
            id next_id = _hpylm->sample_next_token(context_token_ids, _vocab->get_all_token_ids());
            if(next_id == ID_EOS){
                vector<id> token_ids(context_token_ids.begin() + _hpylm->_depth, context_token_ids.end());
                return _vocab->token_ids_to_sentence(token_ids);
            }
            context_token_ids.push_back(next_id);
        }
        return _vocab->token_ids_to_sentence(context_token_ids);
    }

    ////
    void enumerate_phrases_at_depth(int depth, vector<pair<vector<wstring>, double>> &phrases){
        assert(depth <= _hpylm->_depth);
        // Search nodes of specified depth
        vector<Node*> nodes;
        _hpylm->_root->enumerate_nodes_at_depth(depth, nodes);
        for(auto &node: nodes){
            vector<id> tokens;
            vector<wstring> phrase;
            vector<Node*> token_nodes;
            while(node->_parent){
                token_nodes.push_back(node);
                tokens.push_back(node->_token_id);
                phrase.push_back(_vocab->token_id_to_string(node->_token_id));
                node = node->_parent;
            }
            double log_pw = _hpylm->compute_token_log_p(tokens, token_nodes);
            phrases.push_back({phrase, log_pw});
        }
    }

    /// ARPA file generation
    void export_to_arpa_file(string filename)
    {
        class my_numpunct: public std::numpunct<char> {
            std::string do_grouping() const { return ""; }
        };

        int ngram = _hpylm->_depth + 1;
        unordered_map<int, int> map;
        _hpylm->count_tokens_of_each_depth(map);
        //
        std::ofstream fout(filename);
        std::stringstream header;
        std::stringstream buffer;
        header << "\\data\\" << endl;
        //writing n-grams
        for(int i = 1; i < ngram; i++) {
            buffer << endl << "\\" << i << "-grams:" << endl;
            vector<pair<vector<wstring>, double>> phrases;
            enumerate_phrases_at_depth(i, phrases);
            if(i == 1) {
                header << "ngram " << i << "=" << to_string(phrases.size()+2) << endl;
                buffer << "0\t<s>" << endl << "0\t</s>" << endl;
            } else {
                header << "ngram " << i << "=" << to_string(phrases.size()) << endl;
            }
            ///
            for(auto &phrase: phrases) {
                buffer << phrase.second << "\t";
                int c = 0;
                for (wstring word: phrase.first) {
                    string str( word.begin(), word.end() );
                    buffer << str;
                    if (++c % i != 0)
                        buffer << " ";
                }
                buffer << endl;
            }
        }
        fout << header.str();
        fout << buffer.str() << "\\end\\";
        fout.close();
    }
    ///

    template <typename Iterator>
    void sprintf_approach(Iterator begin, Iterator end, const std::string &fileName) {
        std::stringstream buffer;
        toStringStream(begin, end, buffer);
        std::ofstream fout(fileName);
        fout << buffer.str();
        fout.close();
    }

};

static void show_usage(std::string name)
{
    std::cerr << "Generating HPYLM language model\n"
              << "Usage: " << name << " [<option(s)>] text-file.txt\n"
              << "Options:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t-o,--order N\tSpecify the N-gram (order), DEFAULT: 3\n"
              << "\t-a,--arpa FILENAME\tSpecify the ARPA-file name, DEFAULT: model.arpa"
              << std::endl;
}

int main(int argc, char *argv[])
{
    int order = 3;
    string arpa_file = "model.arpa";
    string text_file = "";
    if (argc < 2) {
        show_usage(argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; ++i) {
            string arg = argv[i];
            if ((arg == "-h") || (arg == "--help")) {
                show_usage(argv[0]);
                return 0;
            } else if ((arg == "-o") || (arg == "--order")) {
                if (i + 1 < argc) {
                    order = atoi(argv[++i]);
                } else {
                      cerr << "--order option requires one argument." << endl;
                    return 1;
                }
            }  else if ((arg == "-a") || (arg == "--arpa")) {
                if (i + 1 < argc) {
                    arpa_file = argv[++i];
                } else {
                      cerr << "--arpa-file option requires one argument." << endl;
                    return 1;
                }
            }
            else {
                text_file = argv[i];
            }
    }
    // default file for test purposes
    string dirname = "out";
    if(text_file == "")
        text_file = "dataset/wiki.txt";
    //creating model & parameters
    PyHPYLM *model = new PyHPYLM(order);
    model->load_textfile(text_file, 0.90);
    model->set_g0(1.0 / model->get_num_types_of_words());
    model->perform_gibbs_sampling();
    model->sample_hyperparameters();
    //exporting to arpa-file
    model->export_to_arpa_file(arpa_file);
    //printing model parameters
    printf("# of nodes: %d\n", model->_hpylm->get_num_nodes());
    printf("# of customers: %d\n", model->_hpylm->get_num_customers());
    printf("# of tables: %d\n", model->_hpylm->get_num_tables());
    printf("# Perplexity of train dataset: %f\n", model->compute_perplexity_train());
    printf("# Perplexity of test dataset: %f\n", model->compute_perplexity_test());
    //releasing memory
    delete model;
    cout << "[i] "<< order << "-gram model saved to ARPA-file (" << arpa_file << ")" << endl;
    return 0;
}

#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <string>
#include <unordered_set>
#include <unordered_map>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "cnn/dict.h"
#include "cnn/cfsm-builder.h"

#include "nt-parser/rnn_grammar.h"
#include "nt-parser/pretrained.h"
#include "nt-parser/compressed-fstream.h"
#include "nt-parser/parser_disc.h"

volatile bool requested_stop = false;
unsigned N_SAMPLES = 1;

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;


vector<bool> singletons; // used during training

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("explicit_terminal_reduce,x", "[recommended] If set, the parser must explicitly process a REDUCE operation to complete a preterminal constituent")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("bracketing_dev_data,C", po::value<string>(), "Development bracketed corpus")

        ("test_data,p", po::value<string>(), "Test corpus")
        ("dropout,D", po::value<float>(), "Dropout rate")
        ("samples,s", po::value<unsigned>(), "Sample N trees for each test sentence instead of greedy max decoding")
        ("alpha,a", po::value<float>(), "Flatten (0 < alpha < 1) or sharpen (1 < alpha) sampling distribution")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("split_pos_tags", "split POS tags")
        ("separate_dicts", "split vocabulary between raw/unk and lc")
        ("use_edge_labels", "train and predict function labels")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("beam_size,b", po::value<unsigned>()->default_value(1), "beam size")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

double eval_output(const string out_fname, const string dev_fname) {
  std::string command="python remove_dev_unk.py "+ dev_fname +" "+ out_fname
    + " > evaluable.txt";
  const char* cmd=command.c_str();
  system(cmd);
  
  std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+ dev_fname +
    " evaluable.txt > evalbout.txt";
  const char* cmd2=command2.c_str();
  system(cmd2);
  
  std::ifstream evalfile("evalbout.txt");
  std::string lineS;
  std::string brackstr="Bracketing FMeasure";
  double newfmeasure=0.0;
  std::string strfmeasure="";
  while (getline(evalfile, lineS) && !newfmeasure){
    if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
      //std::cout<<lineS<<"\n";
      strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
      std::string::size_type sz;     // alias of size_t
      
      newfmeasure = std::stod (strfmeasure,&sz);
      //std::cout<<strfmeasure<<"\n";
    }
  }
  return newfmeasure;
}
  

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  RNNGrammar gram;

  cerr << "COMMAND LINE:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  NetworkSettings settings;
  gram.separate_dicts = conf.count("separate_dicts");
  settings.IMPLICIT_REDUCE_AFTER_SHIFT = conf.count("explicit_terminal_reduce") == 0;
  settings.USE_POS = conf.count("use_pos_tags");
  settings.SPLIT_POS = conf.count("split_pos_tags");
  settings.USE_EDGES = conf.count("use_edge_labels");
  if (conf.count("dropout"))
    settings.DROPOUT = conf["dropout"].as<float>();
  settings.LAYERS = conf["layers"].as<unsigned>();
  settings.INPUT_DIM = conf["input_dim"].as<unsigned>();
  settings.PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  settings.HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  settings.ACTION_DIM = conf["action_dim"].as<unsigned>();
  settings.LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  settings.POS_DIM = conf["pos_dim"].as<unsigned>();
  if (conf.count("train") && conf.count("dev_data") == 0) {
    cerr << "You specified --train but did not specify --dev_data FILE\n";
    return 1;
  }
  if (conf.count("alpha")) {
    settings.ALPHA = conf["alpha"].as<float>();
    if (settings.ALPHA <= 0.f) {
        cerr << "--alpha must be between 0 and +infty\n"; abort();
    }
  }
  if (conf.count("samples")) {
    N_SAMPLES = conf["samples"].as<unsigned>();
    if (N_SAMPLES == 0) { cerr << "Please specify N>0 samples\n"; abort(); }
  }
  
  const string fname = settings.propose_filename();
  cerr << "PARAMETER FILE: " << fname << endl;
  bool softlinkCreated = false;

  Model model;
  unordered_map<unsigned, vector<float>> pretrained;

  Corpus corpus(&gram);
  Corpus dev_corpus(&gram);
  Corpus test_corpus(&gram);
  corpus.load_oracle(conf["training_data"].as<string>(), settings.SPLIT_POS);

  if (conf.count("words")) {
    parser::ReadEmbeddings_word2vec(conf["words"].as<string>(),
				    &gram.termdict, &pretrained);
  }

  // freeze dictionaries so we don't accidentaly load OOVs
  gram.Freeze();

  // compute the singletons in the parser's training data
  unordered_map<unsigned, unsigned> counts;
  corpus.count(counts);
  singletons.resize(gram.termdict.size(), false);
  for (auto wc : counts) {
    if (wc.second == 1) singletons[wc.first] = true;
  }

  if (conf.count("dev_data")) {
    cerr << "Loading validation set\n";
    dev_corpus.load_oracle(conf["dev_data"].as<string>(), settings.SPLIT_POS);
  }
  if (conf.count("test_data")) {
    cerr << "Loading test set\n";
    test_corpus.load_oracle(conf["test_data"].as<string>(), settings.SPLIT_POS);
  }

  ParserBuilder parser(&gram, &settings, singletons, &model, &pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer sgd(&model);
    //AdamTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    //sgd.eta_decay = 0.08;
    sgd.eta_decay = 0.05;
    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min((int)status_every_i_iterations, (int)corpus.sents.size());
    unsigned si = corpus.sents.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.sents.size() << endl;
    unsigned trs = 0;
    unsigned words = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    double best_dev_err = 9e99;
    double bestf1=0.0;
    //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
    while(!requested_stop) {
      ++iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.sents.size()) {
             si = 0;
             if (first) { first = false; } else { sgd.update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           auto& sentence = corpus.sents[order[si]];
	   const vector<unsigned>& actions=corpus.actions[order[si]];
           const vector<unsigned>& edge_labels=corpus.edge_labels[order[si]];
           ComputationGraph hg;
           parser.log_prob_parser(&hg, sentence, actions, edge_labels, nullptr,
                   &right,false);
           double lp = as_scalar(hg.incremental_forward());
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward();
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
           words += sentence.size();
      }
      sgd.status();
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<
         /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
        ") per-action-ppl: " << exp(llh / trs) << " per-input-ppl: " << exp(llh / words) << " per-sent-ppl: " << exp(llh / status_every_i_iterations) << " err: " << (trs - right) / trs << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;
      llh = trs = right = words = 0;

      static int logc = 0;
      ++logc;
      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = dev_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        ostringstream os;
        os << "/tmp/parser_dev_eval." << getpid() << ".txt";
        const string pfx = os.str();
        ofstream out(pfx.c_str());
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const auto& sentence=dev_corpus.sents[sii];
	   const vector<unsigned>& actions = dev_corpus.actions[sii];
           const vector<unsigned>& edge_labels = dev_corpus.edge_labels[sii];
           dwords += sentence.size();
           {  ComputationGraph hg;
              parser.log_prob_parser(&hg,sentence,actions,edge_labels,
                      nullptr,&right,true);
              double lp = as_scalar(hg.incremental_forward());
              llh += lp;
           }
           ComputationGraph hg;
           if (settings.USE_EDGES) {
             vector<unsigned> pred_edge;
             vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,
                 vector<unsigned>(),vector<unsigned>(),
                 &pred_edge, &right,true);
             gram.write_tree(out, pred, pred_edge, sentence);
           } else {
             vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,
                 vector<unsigned>(),vector<unsigned>(),
                 nullptr, &right,true);
             gram.write_tree(out, pred, sentence);
           }
           trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;
	double newfmeasure = eval_output(pfx, conf["bracketing_dev_data"].as<string>());
        
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " f1: " << newfmeasure << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
//        if (err < best_dev_err && (tot_seen / corpus.size()) > 1.0) {
       if (newfmeasure>bestf1) {
          cerr << "  new best...writing model to " << fname << " ...\n";
          best_dev_err = err;
	  bestf1=newfmeasure;
          save_model(model, fname);
          system((string("cp ") + pfx + string(" ") + pfx + string(".best")).c_str());
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
        }
      }
    }
  } // should do training?
  if (test_corpus.size() > 0) { // do test evaluation
        bool sample = conf.count("samples") > 0;
        unsigned test_size = test_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        auto t_start = chrono::high_resolution_clock::now();
	const vector<unsigned> actions;
        const vector<unsigned> edge_labels;
        for (unsigned sii = 0; sii < test_size; ++sii) {
          const auto& sentence=test_corpus.sents[sii];
          dwords += sentence.size();
          for (unsigned z = 0; z < N_SAMPLES; ++z) {
            ComputationGraph hg;
            vector<unsigned> pred_edge_labels;
            vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,
                actions, edge_labels, &pred_edge_labels, &right, sample,true);
            double lp = as_scalar(hg.incremental_forward());
            cout << sii << " ||| " << -lp << " |||";
            if (settings.USE_EDGES) {
              gram.write_tree(cout, pred, pred_edge_labels, sentence);
            } else {
              gram.write_tree(cout, pred, sentence);
            }
          }
        }
        ostringstream os;
        os << "/tmp/parser_test_eval." << getpid() << ".txt";
        const string pfx = os.str();
        ofstream out(pfx.c_str());
        t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < test_size; ++sii) {
          const auto& sentence=test_corpus.sents[sii];
          const vector<unsigned>& actions=test_corpus.actions[sii];
          const vector<unsigned>& edge_labels = test_corpus.edge_labels[sii];
          dwords += sentence.size();
          {  ComputationGraph hg;
            parser.log_prob_parser(&hg,sentence,actions,edge_labels,
                nullptr, &right,true);
            double lp = as_scalar(hg.incremental_forward());
            llh += lp;
          }
          ComputationGraph hg;
          vector<unsigned> pred_edges;
          vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,
              vector<unsigned>(), vector<unsigned>(),
              &pred_edges, &right,true);
          if (settings.USE_EDGES) {
            gram.write_tree(out, pred, pred_edges, sentence);
          } else {
            gram.write_tree(out, pred, sentence);
          }
          double lp = 0;
          trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;
        cerr << "Test output in " << pfx << endl;
        //parser::EvalBResults res = parser::Evaluate("foo", pfx);
        std::string command="python remove_dev_unk.py "+ conf["devdata"].as<string>() +" "+pfx+" > evaluable.txt";
        const char* cmd=command.c_str();
        system(cmd);

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+conf["devdata"].as<string>()+" evaluable.txt>evalbout.txt";
        const char* cmd2=command2.c_str();

        system(cmd2);

        std::ifstream evalfile("evalbout.txt");
        std::string lineS;
        std::string brackstr="Bracketing FMeasure";
        double newfmeasure=0.0;
        std::string strfmeasure="";
        while (getline(evalfile, lineS) && !newfmeasure){
                if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
                        //std::cout<<lineS<<"\n";
                        strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;
                        newfmeasure = std::stod (strfmeasure,&sz);
                        //std::cout<<strfmeasure<<"\n";
                }
        }

       cerr<<"F1score: "<<newfmeasure<<"\n";
    
  }
}

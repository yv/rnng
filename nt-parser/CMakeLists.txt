PROJECT(cnn:nt-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(nt-parser nt-parser.cc parser_disc.cc rnn_grammar.cc pretrained.cc)
target_link_libraries(nt-parser cnn ${Boost_LIBRARIES} z)

set_source_files_properties( rnn_python.pyx PROPERTIES CYTHON_IS_CXX 1 )
cython_add_module( rnn_python rnn_python.pyx parser_disc.cc rnn_grammar.cc pretrained.cc)
target_link_libraries(rnn_python cnn_shared ${Boost_LIBRARIES} z)

ADD_EXECUTABLE(nt-parser-gen nt-parser-gen.cc rnn_grammar.cc pretrained.cc)
target_link_libraries(nt-parser-gen cnn ${Boost_LIBRARIES} z)

ADD_EXECUTABLE(test_edges test_edges.cc rnn_grammar.cc)
target_link_libraries(test_edges ${Boost_LIBRARIES} z)

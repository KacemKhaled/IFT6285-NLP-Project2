cmake_minimum_required (VERSION 2.8.0)
project ("ZGen - A collections of sentence linearization tools")

find_package (Boost REQUIRED)

add_definitions (-DBOOST_LOG_DYN_LINK)

set (CMAKE_CXX_FLAGS "-std=c++11 -g")
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (SOURCE_DIR             ${PROJECT_SOURCE_DIR}/src)

include_directories (${Boost_INCLUDE_DIRS} ${SOURCE_DIR})

link_directories (${Boost_LIBRARY_DIRS})

#add_executable (zgen-bf
#    ${SOURCE_DIR}/utils/ioutils.cc
#    ${SOURCE_DIR}/utils/logging.cc
#    ${SOURCE_DIR}/types/instance.cc
#    ${SOURCE_DIR}/knowledge/postag_constrain.cc
#    ${SOURCE_DIR}/engine/postag_alphabet.cc
#    ${SOURCE_DIR}/engine/deprel_alphabet.cc
#    ${SOURCE_DIR}/engine/word_alphabet.cc
#    ${SOURCE_DIR}/bestfirst/learning/learner.cc
#    ${SOURCE_DIR}/bestfirst/model/model.cc
#    ${SOURCE_DIR}/bestfirst/search/span.cc
#    ${SOURCE_DIR}/bestfirst/search/span_utils.cc
#    ${SOURCE_DIR}/bestfirst/search/span_allocator.cc
#    ${SOURCE_DIR}/bestfirst/search/span_factory.cc
#    ${SOURCE_DIR}/bestfirst/pipe/pipe.cc
#    ${SOURCE_DIR}/bestfirst/frontend/opt.cc
#    ${SOURCE_DIR}/bestfirst/frontend/main.cc
#    )

#target_link_libraries (zgen-bf
#    boost_program_options
#    boost_log_setup
#    boost_log
#    boost_regex
#    boost_serialization
#    boost_thread
#    boost_system
#    pthread)

add_executable (ZGen_SR
    ${SOURCE_DIR}/utils/ioutils.cc
    ${SOURCE_DIR}/utils/logging.cc
    ${SOURCE_DIR}/utils/math_utils.cc
    ${SOURCE_DIR}/types/instance.cc
    ${SOURCE_DIR}/types/instance_utils.cc
    ${SOURCE_DIR}/types/tree.cc
    ${SOURCE_DIR}/knowledge/postag_constraint.cc
    ${SOURCE_DIR}/engine/token_alphabet.cc
    ${SOURCE_DIR}/shiftreduce/types/action.cc
    ${SOURCE_DIR}/shiftreduce/types/state.cc
    ${SOURCE_DIR}/shiftreduce/types/action_utils.cc
    ${SOURCE_DIR}/shiftreduce/model/model.cc
    ${SOURCE_DIR}/shiftreduce/learning/learner.cc
    ${SOURCE_DIR}/shiftreduce/pipe/pipe.cc
    ${SOURCE_DIR}/shiftreduce/decode/decode_context.cc
    ${SOURCE_DIR}/shiftreduce/decode/decoder.cc
    ${SOURCE_DIR}/shiftreduce/decode/none_decoder.cc
    ${SOURCE_DIR}/shiftreduce/decode/partial_decoder.cc
    ${SOURCE_DIR}/shiftreduce/decode/full_decoder.cc
    ${SOURCE_DIR}/shiftreduce/frontend/opt.cc
    ${SOURCE_DIR}/shiftreduce/frontend/main.cc
    )

target_link_libraries (ZGen_SR
    boost_program_options
    boost_log_setup
    boost_log
    boost_regex
    boost_serialization
    boost_thread
    boost_system
    pthread)

add_executable (unittest
#    ${SOURCE_DIR}/types/instance.cc
#    ${SOURCE_DIR}/utils/math_utils.cc
#    ${SOURCE_DIR}/utils/ioutils.cc
#    ${SOURCE_DIR}/bestfirst/search/span.cc
#    ${SOURCE_DIR}/bestfirst/search/span_utils.cc
#    ${SOURCE_DIR}/bestfirst/search/span_allocator.cc
    ${SOURCE_DIR}/shiftreduce/types/action.cc
    ${SOURCE_DIR}/unittest/test.cc
    ${SOURCE_DIR}/unittest/shiftreduce/test_action.cc
    ${SOURCE_DIR}/unittest/shiftreduce/test_score.cc
#    ${SOURCE_DIR}/unittest/bestfirst/test_span_allocator.cc
#    ${SOURCE_DIR}/unittest/bestfirst/test_span.cc
#    ${SOURCE_DIR}/unittest/bestfirst/test_span_utils.cc
#    ${SOURCE_DIR}/unittest/bestfirst/test_span_repr.cc
#    ${SOURCE_DIR}/unittest/bestfirst/test_span_types.cc
   )

target_link_libraries (unittest
    boost_program_options
    boost_log_setup
    boost_log
    boost_regex
    boost_serialization
    boost_thread
    boost_system
    boost_unit_test_framework
    pthread)

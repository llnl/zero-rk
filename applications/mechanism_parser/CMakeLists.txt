


add_executable(mechanism_parser.x mechanism_parser.cpp MechanismParserIFP.cpp)

target_link_libraries(mechanism_parser.x zerork zerorkutilities spify)
add_spifyIFP_target(MechanismParserIFP mechanism_parser.x)

install(TARGETS mechanism_parser.x
        RUNTIME DESTINATION bin)


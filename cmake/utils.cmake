function(print_status)
    foreach(var ${ARGN})
        message(STATUS "${var}: ${${var}}")
    endforeach()
endfunction()

function(print_status_env)
    foreach(var ${ARGN})
        message(STATUS "${var}: $ENV{${var}}")
    endforeach()
endfunction()
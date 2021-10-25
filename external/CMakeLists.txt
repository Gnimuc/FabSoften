include(FetchContent)

# dlib's face landmark detection model
FetchContent_Declare(
    shape_predictor_68_face_landmarks
    URL      http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    URL_HASH SHA256=7d6637b8f34ddb0c1363e09a4628acb34314019ec3566fd66b80c04dda6980f5
    DOWNLOAD_NO_EXTRACT true
)

FetchContent_MakeAvailable(shape_predictor_68_face_landmarks)

set(face_model_dir ${shape_predictor_68_face_landmarks_SOURCE_DIR})
set(face_model_archive ${face_model_dir}/shape_predictor_68_face_landmarks.dat.bz2)
set(face_model_srcdir ${CMAKE_SOURCE_DIR}/models)
set(face_model_dat ${face_model_srcdir}/shape_predictor_68_face_landmarks.dat)

if (NOT EXISTS ${face_model_dat})
    message(STATUS "${face_model_dat} is missing. Try to fetch again...")
    if(EXISTS ${face_model_archive})
        message(STATUS "Check downloaded info:")
        print_status(face_model_dir face_model_archive)
        if (WIN32)
            find_program(EXE_7ZIP 7z.exe)
            if (EXE_7ZIP)
                message(STATUS "Found 7zip: ${EXE_7ZIP}")
                # need to use 7zip for extracting .bz2 archives on Windows
                execute_process(
                    COMMAND ${EXE_7ZIP} x ${face_model_archive} -o${face_model_srcdir}
                    WORKING_DIRECTORY ${face_model_dir}
                )
            else()
                message(SEND_ERROR "Cannot find 7zip. Please make sure 7z is avaliable on the command line.")
            endif()
        else()
            # execute_process(
            #     COMMAND ${CMAKE_COMMAND} -E tar xvf ${face_model_archive} -o ${face_model_srcdir}
            #     WORKING_DIRECTORY ${face_model_dir}
            # )
            file(
                ARCHIVE_EXTRACT
                INPUT ${face_model_archive}
                DESTINATION ${face_model_srcdir}
            )
        endif()
    endif()
else()
    message(STATUS "Found ${face_model_dat}")
endif()

if(NOT EXISTS ${face_model_dat})
    message(FATAL_ERROR "The dlib face landmark detection model is missing.")
endif()


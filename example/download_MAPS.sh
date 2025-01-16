#!/bin/bash

# This script is heavily based off the ALMA archive download script.

# This script runs on Linux and MaxOS and downloads all the selected files to the current working directory in up to 5 parallel download streams.
# Should a download be aborted just run the entire script again, as partial downloads will be resumed. Please play nice with the download systems
# at the ARCs and do not increase the number of parallel streams.

# connect / read timeout for wget / curl
export TIMEOUT_SECS=300
# how many times do we want to automatically resume an interrupted download?
export MAX_RETRIES=3
# after a timeout, before we retry, wait a bit. Maybe the servers were overloaded, or there was some scheduled downtime.
# with the default settings we have 15 minutes to bring the dataportal service back up.
export WAIT_SECS_BEFORE_RETRY=300
# the files to be downloaded
LIST=("
ftp://ftp.cv.nrao.edu/NRAO-staff/rloomis/MAPS/MWC_480/images/CO/robust_0.5/MWC_480_CO_220GHz.robust_0.5.JvMcorr.image.pbcor.fits
")

# If we terminate the script using CTRL-C during parallel downloads, the remainder of the script is executed, asking if
# the user wants to unpack tar files. Not very nice. Exit the whole script when the user hits CTRL-C.
trap "exit" INT


export failed_downloads=0

# download a single file.
# attempt the download up to N times
function dl {
  local file=$1
  local filename=$(basename $file)
  # the nth attempt to download a single file
  local attempt_num=0

  # wait for some time before starting - this is to stagger the load on the server (download start-up is relatively expensive)
  sleep $[ ( $RANDOM % 10 ) + 2 ]s

  if command -v "curl" > /dev/null 2>&1; then
    local tool_name="curl"
    local download_command=(curl -S -s -k -O -f --speed-limit 1 --speed-time ${TIMEOUT_SECS})
  elif command -v "wget" > /dev/null 2>&1; then
    local tool_name="wget"
    local download_command=(wget -c -q -nv --timeout=${TIMEOUT_SECS} --tries=1)
  fi

  # manually retry downloads. 
  # I know wget and curl can both do this, but curl (as of 10.04.2018) will not allow retry and resume. I want consistent behaviour so 
  # we implement the retry mechanism ourselves.
  echo "starting download of $filename"
  until [ ${attempt_num} -ge ${MAX_RETRIES} ]
  do
    # echo "${download_command[@]}" "$file"
    $("${download_command[@]}" "$file")
    status=$?
    # echo "status ${status}"
    if [ ${status} -eq 0 ]; then
      echo "	    succesfully downloaded $filename"
      break
    else
      failed_downloads=1
      echo "		download $filename was interrupted with error code ${tool_name}/${status}"
      attempt_num=$[${attempt_num}+1]
      if [ ${attempt_num} -ge ${MAX_RETRIES} ]; then
        echo "	  ERROR giving up on downloading $filename after ${MAX_RETRIES} attempts  - rerun the script manually to retry."
      else
        echo "		download $filename will automatically resume after ${WAIT_SECS_BEFORE_RETRY} seconds"
        sleep ${WAIT_SECS_BEFORE_RETRY}
        echo "		resuming download of $filename, attempt $[${attempt_num}+1]"
      fi
    fi
  done
}
export -f dl


# temporary workaround for ICT-13558: "xargs -I {}" fails on macos with variable substitution where the length of the variable
# is greater than 255 characters. For the moment we download these long filenames in serial. At some point I'll address this issue
# properly, allowing parallel downloads.
# Array of filenames for download where the filename > 251 characters
# 251? Yes. The argument passed to bash is "dl FILENAME;" In total it cannot exceed 255. So FILENAME can only be 251
export long_files=()
# arrayf of filenames with length <= 255 characters - can be downloaded in parallel.
export ok_files=()
function split_files_list {
	for nextfile in ${LIST}; do
		length=${#nextfile}
		if [[ $length -ge 251 ]]; then
			long_files+=($nextfile)
		else
			ok_files+=($nextfile)
		fi
	done
}

# Main body
# ---------

# check that we have one of the required download tools
if ! (command -v "wget" > /dev/null 2>&1 || command -v "curl" > /dev/null 2>&1); then
   echo "ERROR: neither 'wget' nor 'curl' are available on your computer. Please install one of them.";
   exit 1
fi

echo "Downloading the following files in up to 5 parallel streams. Total size is 2.08 GB."
echo "${LIST}"
echo "In case of errors each download will be automatically resumed up to 3 times after a 5 minute delay"
echo "To manually resume interrupted downloads just re-run the script."
# tr converts spaces into newlines. Written legibly (spaces replaced by '_') we have: tr "\_"_"\\n"
# IMPORTANT. Please do not increase the parallelism. This may result in your downloads being throttled.
# Please do not split downloads of a single file into multiple parallel pieces.

echo "your downloads will start shortly...."
split_files_list
# "dl" is a function for downloading. I abbreviated the name to leave more space for the filename
echo ${ok_files[@]} | tr \  \\n | xargs -P5 -n1 -I '{}' bash -c 'dl {};'
for next_file in ${long_files[@]}; do
	dl ${next_file}
done

echo "Done."

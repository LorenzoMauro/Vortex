
echoRed() {
    echo -e "\e[31m$1\e[0m" >&2
}

echoGreen() {
    echo -e "\e[32m$1\e[0m" >&2
}

# Function to ask yes or no question using vbscript
askYesOrNo() {
    local message=$1
    echo "Set objShell = CreateObject(\"WScript.Shell\")" > tmpPopup.vbs
    echo "intAnswer = objShell.Popup(\"$message\", 0, \"Query\", 3 + 32)" >> tmpPopup.vbs
    echo "WScript.Echo intAnswer" >> tmpPopup.vbs
    local result=$(cscript //nologo tmpPopup.vbs)
    rm tmpPopup.vbs
    echo "$result"
}

askForPath() {
    local customMessage=$1
    echo "Set shell = CreateObject(\"Shell.Application\")" > tmpFolderPicker.vbs
    echo "Set folder = shell.BrowseForFolder(0, \"$customMessage\", 0)" >> tmpFolderPicker.vbs
    echo "If Not folder is Nothing Then" >> tmpFolderPicker.vbs
    echo "  WScript.Echo folder.Self.Path" >> tmpFolderPicker.vbs
    echo "End If" >> tmpFolderPicker.vbs
    local folder=$(cscript //nologo tmpFolderPicker.vbs)
    rm tmpFolderPicker.vbs
    echo "$folder"
}

# Function to check if a step has been done before
stepDoneBefore() {
    local stepName=$1
    grep -q "$stepName" $logFile 2>/dev/null
}

# Function to mark a step as done
markStepDone() {
    local stepName=$1
    echo "$stepName" >> $logFile
}

# Function to read a value based on key from logFile
readValueFromLog() {
    local key=$1
    grep "^$key=" $logFile | cut -d '=' -f 2-
}

# Function to check, ask, and log steps
checkAndLogStep() {
    local stepKey=$1
    local logKey=$2
    local defaultPath=$3
    local promptMessage=$4

    local stepPath

    if ! stepDoneBefore "$stepKey"; then
        stepPath=$(readValueFromLog "$logKey")
        if [ -z "$stepPath" ]; then
            stepPath="$defaultPath"
        fi
        if [ ! -d "$stepPath" ]; then
            stepPath=$(askForPath "$promptMessage")
        fi
        echo "$logKey located in ${stepPath}" >&2
        echo "$logKey=$stepPath" >> $logFile
        markStepDone "$stepKey"
    else
        stepPath=$(readValueFromLog "$logKey")
        echoGreen "$logKey located in ${stepPath}"
    fi

    echo "$stepPath"
}

SET conda_env_name=windows_build_ns
SET python_version=3.10
cd ../../..

FOR /F %%i IN ('conda info -e ^| find /c "%conda_env_name%"') do SET CONDA_COUNT=%%i
if %CONDA_COUNT% EQU 0 (
    CALL conda create python=%python_version% -y -n %conda_env_name%
)

IF %ERRORLEVEL% NEQ 0 (
    echo "Could not create new conda environment."
    exit 1
)
CALL conda activate %conda_env_name%
CALL pip uninstall neural-speed -y
echo "pip list all the components------------->"
CALL pip list
CALL pip install -U pip --proxy=proxy-prc.intel.com:913
echo "Installing requirements for validation scripts..."
CALL pip install -r requirements.txt --proxy=proxy-prc.intel.com:913
echo "pip list all the components------------->"
CALL pip list
echo "------------------------------------------"
IF %ERRORLEVEL% NEQ 0 (
    echo "Could not install requirements."
    exit 1
)

git submodule update --init --recursive
python setup.py sdist bdist_wheel
IF %ERRORLEVEL% NEQ 0 (
    echo "Could not build binary."
    exit 1
)

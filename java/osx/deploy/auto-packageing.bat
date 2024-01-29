@echo off
if "%1"==":batch" goto :batch_start
start /b cmd /c "%~dpnx0" :batch
exit /b
:batch_start
setlocal

set "pwd=%cd%"
set "cwd=%~dp0"
cd /d "%cwd%"

rmdir /s /q osx 2>nul
if not exist "osx" (
    mkdir osx
)
if not exist "osx\lib" (
    mkdir osx\lib
)
if not exist "osx\pb_lib" (
    mkdir osx\pb_lib
)
if not exist "osx\conf" (
    mkdir osx\conf
)
if not exist "osx\extension" (
    mkdir osx\extension
)
if not exist "osx\conf\broker" (
    mkdir osx\conf\broker
)
if not exist "osx\conf\components" (
    mkdir osx\conf\components
)

cd ..
call mvn clean package -DskipTests

if not exist "lib" (
    mkdir lib
)

xcopy /y osx-broker\target\*.jar deploy\osx\lib\
xcopy /y osx-pb-v2\target\*.jar deploy\osx\lib\pb_lib
xcopy /y osx-pb-v3\target\*.jar deploy\osx\lib\pb_lib
xcopy /y /e osx-broker\target\lib\* deploy\osx\lib\
xcopy /y osx-broker\src\main\resources\broker\* deploy\osx\conf\broker\
xcopy /y /e osx-broker\src\main\resources\components\* deploy\osx\conf\components\
copy bin\service.sh deploy\osx\
start powershell -Command "(Get-Content deploy\osx\service.sh) -replace '\r', '' | Set-Content deploy\osx\service.sh"
cd deploy
tar -czf osx.tar.gz osx

cd /d "%pwd%"

endlocal

echo package successful
pause
exit /b 0
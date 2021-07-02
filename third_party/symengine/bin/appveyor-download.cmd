@echo off
rem initiate the retry number
set retryNumber=0
set maxRetries=6
set time=0

:DOWNLOAD
timeout %time% > NUL
set /a time=2*%time%+1
appveyor DownloadFile %*

rem problem?
IF NOT ERRORLEVEL 1 GOTO :EOF
@echo Oops, appveyor download exited with code %ERRORLEVEL% - let us try again!
set /a retryNumber=%retryNumber%+1
IF %reTryNumber% LSS %maxRetries% (GOTO :DOWNLOAD)
@echo Sorry, we tried downloading the package for %maxRetries% times and all attempts were unsuccessful!
EXIT /B 1

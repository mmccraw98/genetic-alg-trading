@echo off

:BEGIN
set /p default_yn="Use default parameters? (y/n). . ."
echo %default_yn%
cls

if %default_yn% == y (

	set fsize=10
	set usize=5
	set tlow=3
	set tup=50
	set numgen=5
	set hotstart=None

	echo Enter a Training ID:
	set /p dir=""
	
	goto :START

) else if %default_yn% == n (goto :HOTSTART) else (
	echo Missunderstood entry: %default_yn%, please enter y or n. . .
	goto :BEGIN
)

:BEGIN

:START
echo %fsize%


pause
@echo off

set use_hs_yn=n

:BEGIN
set /p default_yn="Use default parameters? (y/n). . ."
echo %default_yn%
cls

if %default_yn% == y (

	set param1=10
	set param2=5
	set param3=3
	set param4=50
	set param5=5
	set param7=None

	echo Working Directory:
	set /p param6=""
	
	goto :START

) else if %default_yn% == n (goto :HOTSTART) else (
	echo Missunderstood entry: %default_yn%, please enter y or n. . .
	goto :BEGIN
)

:HOTSTART
set /p use_hs_yn="Use hot start launch protocol? (y/n). . ."
echo %use_hs_yn%
cls

if %use_hs_yn% == y (
	
	echo Enter directory for hot start protocol:
	set /p param7=""
	cls
	goto :CHOOSE

) else if %use_hs_yn% == n (
	
	set param7=None
	cls
	goto :CHOOSE

) else (
	echo Missunderstood entry: %use_hs_yn%, please entry y or n. . .
	goto :HOTSTART
)

:CHOOSE
echo Configure Parameters! F-Size: [] U-Size: [] T-Low: [] T-Up: [] Num-Gen: [] Dir: []
echo Forest Size:
set /p param1=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [] T-Low: [] T-Up: [] Num-Gen: [] Dir: []

echo Universe Size:
set /p param2=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [%param2%] T-Low: [] T-Up: [] Num-Gen: [] Dir: []

echo Tree Size - Lower Bound:
set /p param3=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [%param2%] T-Low: [%param3%] T-Up: [] Num-Gen: [] Dir: []

echo Tree Size - Upper Bound:
set /p param4=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [%param2%] T-Low: [%param3%] T-Up: [%param4%] Num-Gen: [] Dir: []

echo Number of Generations to Simulate:
set /p param5=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [%param2%] T-Low: [%param3%] T-Up: [%param4%] Num-Gen: [%param5%] Dir: []

echo Working Directory:
set /p param6=""

:START
cls
echo Summary:
echo A forest of %param1% trees, each with sizes between %param3% and %param4% nodes, will be trained on %param2% randomized 
echo assets for %param5% generations.
echo The intermediate and final results will be saved to: %param6%
if %use_hs_yn% == y (
	
	echo Hot start protocol will be used, drawing models from: %param7%

)

pause

"C:\Users\PC\PycharmProjects\untitled\venv\Scripts\python.exe" "C:\Users\PC\PycharmProjects\Finance\venv\initialize_training.py" --forestsize %param1% --universesize %param2% --treelower %param3% --treeupper %param4% --numgen %param5% --dir %param6% --hot_start %param7%


set /a count = %param5%
for /l %%x in (1, 1, %count%) do (
	"C:\Users\PC\PycharmProjects\untitled\venv\Scripts\python.exe" "C:\Users\PC\PycharmProjects\Finance\venv\train_generations.py" --dir %param6% --iter %%x
)

pause
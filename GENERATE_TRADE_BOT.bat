@echo off

set /p default_yn="Use default parameters? y/n..."
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

) else (


set /p hs_yn="Use hot start? y/n..."
echo %hs_yn%
cls

if %hs_yn% == y (

echo Hot Start Path:
set /p param7=""

echo Hot starting from %param7%.

) else (
echo Configure Parameters! F-Size: [] U-Size: [] T-Low: [] T-Up: [] Num-Gen: [] Dir: [] Hot Start: []
echo Forest Size:
set /p param1=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [] T-Low: [] T-Up: [] Num-Gen: [] Dir: [] Hot Start: []

echo Universe Size:
set /p param2=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [%param2%] T-Low: [] T-Up: [] Num-Gen: [] Dir: [] Hot Start: []

echo Tree Size - Lower Bound:
set /p param3=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [%param2%] T-Low: [%param3%] T-Up: [] Num-Gen: [] Dir: [] Hot Start: []

echo Tree Size - Upper Bound:
set /p param4=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [%param2%] T-Low: [%param3%] T-Up: [%param4%] Num-Gen: [] Dir: [] Hot Start: []

echo Number of Generations to Simulate:
set /p param5=""

cls
echo Configure Parameters! F-Size: [%param1%] U-Size: [%param2%] T-Low: [%param3%] T-Up: [%param4%] Num-Gen: [%param5%] Dir: [] Hot Start: []

echo Working Directory:
set /p param6=""

cls
echo Summary:
echo A forest of %param1% trees, each with sizes between %param3% and %param4% nodes, will be trained on %param2% randomized 
echo assets for %param5% generations.  
)
)

echo The intermediate and final results will be saved to: %param6%

pause


"C:\Users\PC\PycharmProjects\untitled\venv\Scripts\python.exe" "C:\Users\PC\PycharmProjects\Finance\venv\initialize_training.py" --forestsize %param1% --universesize %param2% --treelower %param3% --treeupper %param4% --numgen %param5% --dir %param6% --hot_start %param7%
pause


echo Training Commenced
pause


set /a count = %param5%
for /l %%x in (1, 1, %count%) do (
	"C:\Users\PC\PycharmProjects\untitled\venv\Scripts\python.exe" "C:\Users\PC\PycharmProjects\Finance\venv\train_generations.py" --dir %param6% --iter %%x
)

pause
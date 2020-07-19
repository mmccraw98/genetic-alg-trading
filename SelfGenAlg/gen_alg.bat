@echo off

set use_hs_yn=n
set specifiy_uni_yn=n

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

	echo Enter a Training ID:
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
	
	echo Enter a Training ID for hot start protocol: (name of file from which pre-existing models are drawn)
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

echo Please enter parameters:
echo Enter a Training ID: (will serve as identifying label for generated models and statistics)
set /p param6=""
cls

echo Please enter parameters:
echo Define the number of models per generation: (forest size)
set /p param1=""
cls

echo Please enter parameters:
echo Define the LOWER limit on model size: (min. number of nodes)
set /p param3=""
cls

echo Please enter parameters:
echo Define the UPPER limit on model size: (max. number of nodes)
set /p param4=""
cls

echo Please enter parameters:
echo Define the number of assets in the training environment: (universe size)
set /p param2=""
cls

echo Please enter parameters:
echo Define the number of generations to simulate: 
set /p param5=""
cls

:START

:SPECIFIC
set /p specifiy_uni_yn="Specify assets in training environment? (y/n). . ."
echo %specifiy_uni_yn%
cls

if %specifiy_uni_yn% == y (
	echo Please enter directory containing the desired assets for training:
	set /p %param8%=""
	cls
) else if %specifiy_uni_yn% == n (
	echo The group of assets will be randomly drawn from the financial database
	set %param8%=None
	cls
) else (
	echo Missunderstood entry: %specifiy_uni_yn%, please entry y or n. . .
	goto :SPECIFIC
)
cls


echo Summary:
echo A forest of %param1% trees, each with sizes between %param3% and %param4% nodes, will be trained on %param2% randomized 
echo assets for %param5% generations.
echo The intermediate and final results will be saved under the Training ID: %param6%
if %use_hs_yn% == y (
	
	echo Hot start protocol will be used, drawing models from the Training ID: %param7%

)

echo %param8%

pause

"C:\Users\PC\PycharmProjects\untitled\venv\Scripts\python.exe" "C:\Users\PC\PycharmProjects\SelfGenAlg\initialize_training.py" --forestsize %param1% --universesize %param2% --treelower %param3% --treeupper %param4% --numgen %param5% --dir %param6% --hot_start %param7% --specify %param8%


::set /a count = %param5%
::for /l %%x in (1, 1, %count%) do (
::	"C:\Users\PC\PycharmProjects\untitled\venv\Scripts\python.exe" "C:\Users\PC\PycharmProjects\SelfGenAlg\train_generations.py" --dir %param6% --iter %%x
::)

pause
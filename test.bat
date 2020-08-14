@echo off

echo				____________________________________________________________
echo			       [							    ]
echo			       [Evolution-Driven Trading Algorithm Generation System (EDTGS)]
echo			       [		   Version 1.0.0 - 2020			    ]
echo			       [____________________________________________________________]

pause
cls

echo Please enter a name for this training session.
set /p dir="> "
echo Please enter the number of generations to run.
set /p numgen="> "

"C:\Users\PC\PycharmProjects\untitled\venv\Scripts\python.exe" "C:\Users\PC\PycharmProjects\SelfGenAlg\initialize.py" --dir %dir%

for /l %%x in (1, 1, %numgen%) do (
	"C:\Users\PC\PycharmProjects\untitled\venv\Scripts\python.exe" "C:\Users\PC\PycharmProjects\SelfGenAlg\train.py" --dir %dir%
)

pause
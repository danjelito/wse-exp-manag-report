# Experience Management Report

## Data Source:

1. Class attendance data: From Coco Class Session and Attendance program.
2. Class session data: From Coco Class Session and Attendance program.
3. Non-coco class tracker: A file that is used to record all classes that are not recorded in Coco.
4. Coco trainer data: Consists of teacher data which includes center and working days. Located in Coco Class Session and Attendance program.
5. Coco member: From Coco Member Processor program.
5. Erwin member: From Erwin Member Processor program.
6. Member population: Manual fill, from Coco PowerBI.

## How to Use:

1. Process attendance and session data with Coco Class Session and Attendance program.
2. Complete Non-coco class tracker for that month (Lintang's job) then copy the result into local file.
3. Complete the coco trainer data for that month (ask TO for trainer working days).
4. Process Coco and Erwin member with each processor.
5. Complete Member Population report.
6. IMPORTANT : Set the path to data in .dotenv file.
7. Set the configuration on config.py.
8. Run main.ipynb. This will output full exp management report file.

## Usage:

1. The output of this processor is to be copied into Experience Management Report in google drive.

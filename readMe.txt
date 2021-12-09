How to run my codes:
1. run: pip install -r requirements.txt
2. run: python tiktok.py

How to deploy spark
1. upload the WordVec.py into ec2
    Making sure that trending_sub.csv is in the same directory
2. run: python WordVec.py

Notice1:
We used flask to help build the user interface. 
Due to limited time, we did not deploy the app on web, but all the functions would work just fine.
Therefore, after running tiktok.py, you should click the link it has presented and run the app on localhost.

Notice2:
Concerning how to deal with non-textual data, we have extracted video meta data like video_length, video_size, video_author, video_dpi, etc. 
and added these meta data into our dataset for analysis.
Therefore, we use meta data extracted from videos(non-textual data) for later analysis instead of creating user interface for users to deal with non-textual data.

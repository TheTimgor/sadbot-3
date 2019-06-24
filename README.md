# Installation instructions

*Instructions should work on all operating systems*

1. Install required packages with `pip install -r requirements.txt`
2. This program requires some of the Stanford NLP Group's wonderful tools. As per licensing issues and for the sake of file size, they could not be bundled with the program. You will need to download them.
    * [Stanford Parser core](https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip)
    * [Stanford Parser English models](https://nlp.stanford.edu/software/stanford-english-corenlp-2018-10-05-models.jar)
    * [Stanford Named Entity Recognizer](https://nlp.stanford.edu/software/stanford-ner-2018-10-16.zip)
3. Create a directory named `nlp` in your installation directory
4. Unzip `stanford-parser-full-2018-10-17.zip` and `stanford-ner-2018-10-16.zip` and place them in `nlp/`
5. Place `stanford-english-corenlp-2018-10-05-models.jar` in `/nlp`
6. Open `stanford-english-corenlp-2018-10-05-models.jar` in an archive manager. 
   Extract `edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz` and place it in `nlp/`
7. Make the file `config.json`
    ```json5
    {
        "prefix": "s!",
        "token": "<YOUR TOKEN HERE>",
        "history_backlog": 1000,
        "train_channel": 000000000000000001 // some channel id
    }
    ```
    Alternatively, you can set these all using keyword arguments when running `python3 main.py <token> <prefix> <history backlog> <train channel>`

 
   


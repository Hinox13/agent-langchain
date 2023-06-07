# Agent-langchain
Docker with the Labai tutorial on [langchain agents](https://lablab.ai/t/agents-retrieval-chatbot) expanded with wolfram API. 

The docker files are mounted so is easy to modify and test the code while running the container.


## Set up
- First paste the openai and wolfram API keys in the src/keys.env file.

- The rest is prepared so you just need to build the container and run it. So go to the terminal and do:
```
docker compose build
docker compose up
```

- After doing that the UI starts on port 8000. You can access it by tiping [localhost:8000](http://localhost:8000)

## Play around
That's all! You have your docker set up and can start playing around with langchain agents. 

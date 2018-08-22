#Hunter S. Thompson Bot - char RNN
Here I implemented a char-RNN on the works of Hunter S. Thompson.

## Preprocessing
I took all the works that I could find and save them all into one file, input.txt,
while cleaning it superficially by hand. All letters are converted to lowercase
and some special characters I removed to limit the size of the "vocabulary" to
54. I set this up as a sequence classification task. I would take sequences that
were 100-characters long as the training example and the whatever the next character
was as the training label. Given this 100-character sentence, the network will
attempt to classify it as belonging to 1 of the 54 possible characters.

## Training
I used keras for building the network and training. The input text is 3.1MB, so
not that big. First, I went with a 2-layer LSTM with 256 units in each layer and
dropout of 50% between every layer. After ~100 epochs I was not satisfied with
the result so I defined a new network with 512 units in each layer. After 100 epochs
the result was better but not super convincing. I then changed the dropout from
50% to 10% and found a drastic improvement. I think the high dropout was preventing
the network from fully learning words, and instead just learning chunks of letters
that are used often in English but are not actual words. You can check out more
details in the notebook. I ran everything on an NVIDIA Teslal K40C GPU. Each
epoch with the 512-units per layer network took 90 minutes. So every 100 epochs
took ~ 1 week. For this reason I definitely recommend using checkpoints that save
the best network so far and can spit some nice metrics on the training process.

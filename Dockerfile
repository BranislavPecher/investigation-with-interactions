FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN mkdir project

WORKDIR /project/

ENV PYTHONPATH /project

COPY ./requirements.txt /project/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download(['stopwords', 'punkt', 'wordnet', 'vader_lexicon', 'averaged_perceptron_tagger'])"

CMD ["bash"]

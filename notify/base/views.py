from django.shortcuts import render, redirect
from django.urls import reverse

from django.views.generic import TemplateView


import pyaudio
import wave
import speech_recognition as sr
from django.conf.urls.static import static


from django.core.files.storage import default_storage


from transformers import pipeline,AutoTokenizer,AutoModelForSeq2SeqLM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English
import numpy as np
import requests
import itertools
import torch

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments


API_URL = "https://api-inference.huggingface.co/models/oliverguhr/fullstop-punctuation-multilang-large"
headers = {"Authorization": "Bearer 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"}

#helper functions
async def transcript(file):
    r = sr.Recognizer()
    with sr.AudioFile('media/'+str(file.name)) as source:
        audio_data = r.record(source)
    text = r.recognize_google(audio_data)
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({"inputs":text})
    t=[]
    for a in output:
        if(len(t)>0 and t[-1] == ".\n" ):
            t.append(a['word'].capitalize())
        else:
            t.append(a['word'])
        if(a['entity_group']=='0'):
            t.append(' ')
        elif(a['entity_group']=='.'):
            t.append(a['entity_group']+'\n')
        else:
            t.append(a['entity_group'])

        
    text = ''.join(t)
    text_file = open("media/transcripts/transcript.txt", "w")
    text_file.write(text)
    text_file.close()

def max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset

def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


async def test_train(train_file_path,model,tokenizer,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
 
  train_dataset = load_dataset(train_file_path, tokenizer)
  data_collator = load_data_collator(tokenizer)

  tokenizer.save_pretrained(output_dir)
      

  model.save_pretrained(output_dir)

  training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
      )

  trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
  )
      
  trainer.train()
  trainer.save_model()

def generate_text(model,tokenizer,sequence, max_length):
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

#views
def home(request):
    return render(request,'home.html')


async def speech_to_text(request):
    if request.method == 'GET':
        return render(request,'speech_to_text.html')

    if request.method == 'POST':
        file = request.FILES.get('audio')
        default_storage.save(file.name,file)
        await transcript(file)  
        default_storage.delete(file.name)
        return redirect(reverse('base:summarise'))

def summarise(request):
    #abstractive summary
    tokenizer_cus = AutoTokenizer.from_pretrained("models/abstract_summary/checkpoint-1500")
    model_cus = AutoModelForSeq2SeqLM.from_pretrained("models/abstract_summary/checkpoint-1500")
    summ=pipeline('summarization',model=model_cus,tokenizer=tokenizer_cus)
    text_file = open("media/transcripts/transcript.txt", "r",encoding="utf-8")
    txt = text_file.read()
    text_file.close()
    summary=summ(txt,max_length=int(len(txt)/2))
    text_file = open("media/summaries/summary.txt", "w",encoding="utf-8")
    abs_summarized_text = summary[0].get('summary_text')
    text_file.write(abs_summarized_text)
    text_file.close()

    #extractive summary
    nlp = English()
    nlp.add_pipe('sentencizer')
    doc = nlp(txt.replace("\n", ""))
    sentences = [sent.text.strip() for sent in doc.sents]
    sentence_organizer = {k:v for v,k in enumerate(sentences)}
    tf_idf_vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                    strip_accents='unicode', 
                                    analyzer='word',
                                    token_pattern=r'\w{1,}',
                                    ngram_range=(1, 3), 
                                    use_idf=1,smooth_idf=1,
                                    sublinear_tf=1,
                                    stop_words = 'english')
    tf_idf_vectorizer.fit(sentences)
    sentence_vectors = tf_idf_vectorizer.transform(sentences)
    sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    N = 5
    top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
    mapped_top_n_sentences = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
    mapped_top_n_sentences = sorted(mapped_top_n_sentences, key = lambda x: x[1])
    ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
    ext_summarized_text = " ".join(ordered_scored_sentences)
    params = {'abstract_summary':abs_summarized_text,'extract_summary':ext_summarized_text,'text':txt}
    return render(request,'summarise.html',params)

def keywords(request):
    n_gram_range = (3, 3)
    stop_words = "english"
    text_file = open("media/transcripts/transcript.txt", "r",encoding="utf-8")
    txt = text_file.read()
    text_file.close()
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([txt])
    candidates = count.get_feature_names_out()
    model= torch.load('models/keyword_extraction/keyword_model')
    doc_embedding = model.encode([txt])
    candidate_embeddings = model.encode(candidates)
    keywords = max_sum_sim(doc_embedding,candidate_embeddings, candidates, top_n = 5 , nr_candidates = 20)
    params = {'keywords':keywords}
    return render(request,'keywords.html',params)



async def text_generation(request):
    #device = torch.device("cuda")
    model = load_model("models/text_generation/gpt2_model_configs")
    #model.to(device)
    tokenizer = load_tokenizer("models/text_generation/gpt2_model_configs")

    train_file_path = "media/transcripts/transcript.txt"
    output_dir = 'models/text_generation/gpt2_model_configs'
    overwrite_output_dir = False
    per_device_train_batch_size = 8
    num_train_epochs = 1.0
    save_steps = 500

    

    # await test_train(train_file_path=train_file_path,
    # model=model,
    # tokenizer = tokenizer,
    # output_dir=output_dir,
    # overwrite_output_dir=overwrite_output_dir,
    # per_device_train_batch_size=per_device_train_batch_size,
    # num_train_epochs=num_train_epochs,
    # save_steps=save_steps)



    torch.cuda.empty_cache()
    model = load_model("models/text_generation/gpt2_model_configs")
    tokenizer = load_tokenizer("models/text_generation/gpt2_model_configs")

    generator=pipeline('text-generation',model=model,tokenizer=tokenizer)
    sequence = "Centre for Disease Control and Prevention" 
    #sequence.to(device)
    #max_len = 100
    generated_text = generator(sequence, max_length=100, num_return_sequences=1) 
    torch.cuda.empty_cache()
    params = {"generated_text": generated_text}
    return render(request,'textGenerate.html',params)

    
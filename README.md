# Dialogue System Implementation

This repository houses the implementation of a dialogue system I developed as part of my Master's degree dissertation in Natural Language Processing (NLP).

## Motivation

The primary motivation behind this project was to investigate the significant influence of a speaker's intention on the progression and dynamics of a conversation.

## Framework and Model

- **Framework**: PyTorch
- **Models**:
  - Bidirectional Gated Recurrent Unit (GRU)
  - Linear Chain Conditional Random Field (CRF)

## Preprocessing Pipeline

1. **Tokenization**: Breaking down text into individual words or tokens.
2. **Replacement**: Substituting specific words or characters with predefined ones.
3. **Padding**: Standardizing sentences or sequences to a uniform length.
4. **Numerization**: 
   - **Word2Vec**: Employed for converting words into numerical vectors.
   - **Char2Vec**: Utilized for managing unknown words.
5. **Compressing**: For lengthy sentences, we compressed them in a consistent manner.

## Key Insights

From our experiments, we derived:

- The intention and emotion of the speaker play a pivotal role in determining the flow and direction of the conversation.
- The impact of the speaker's intention on the conversation's progression becomes increasingly noticeable as the dialogue extends.


## Acknowledgements

probability theory, machine learning


https://library.korea.ac.kr/detail/?cid=CAT000045999362&ctype=t



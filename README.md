# **Text-Processing**

## **목차**
1. [텍스트 처리의 단계](#1-텍스트-처리의-단계)
2. [토큰화](#2-토큰화)
3. [문장 토큰화](#3-문장-토큰화)
4. [단어 임베딩](#4-단어-임베딩)
5. [원핫 벡터](#5-원핫벡터one-hot-vector)
6. [이진벡터](#6-이진-벡터)
7. [TF-IDF벡터/코사인 유사도/정규화](#7-tf-idf-벡터)
8. [사이킷런](#8-사이킷런-활용)

---

## 1. 텍스트 처리의 단계
 1. 텍스트 데이터 수집 
    - 자료수집 -> txt, cell, csv 저장
 2. 문자열 토큰화
    - 토큰화 규칙으로 의미 단위 분리
    - 구두점, 특수기호 정보 불필요시 이 과정에서 제거
 3. 불용어(Stopwords) 제거
    - 통계적 의미 없는 단어 제거 (to,the...)
 4. 어간 추출(Stemming)
    - 과거, 현재 복수형등의 변화
    - 정규표현식에서 Stemming과정 진행
 5. 품사(Pos) 태깅
    - 품사 지정(품사 정보가 필요한 경우)
 6. 의미구조(Vector, Matrix) 생성 - Embedding
    - 통계 기반, 추론 기반 시스템
    - 벡터 행렬로 변경
<div style="text-align: right">

[목차](#목차)
</div>
---

## 2. 토큰화

### 분할(Segmentation)
- 정해진 기준으로 하위 개념 정보로 데이터를 분할
    > 문서 -> 문단/문장 단위
    > 문단 -> 문장/ 문장-> 개별 단어

### 토큰/단어
토큰 = 텍스트 분석 단위
단어 = 개념을 나타내는 기호

### 토큰화 방법
1. str.split()

    <details>
    <summary>코드</summary>  

    1. 
    ```python
    sent = 'Thomas Jefferson began building Monticello age of 26.'
    token = sent.split()
    print(token)
    # 결과
    ['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'age', 'of', '26.']
    ```
    2. 
    ```python
    sent = 'Thomas Jefferson began building Monticello age of 26.'
    sent = sent.lower()
    sent = sent.replace('.', ' .')
    token = sent.split()
    print(token)
    
    # 실행결과
    ['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'age', 'of', '26.' , '.']
    ```
    </details>

2. re.split() (정규표현식)
    <details>
    <summary>코드</summary>

    ```python
    import re #정규표현식 모듈 임포트
    
    sent = "Hello!! Python. Coding,   Programming, Study.."
    delim = re.compile(r'[-\s.,!@;?]+') #정규표현식 컴파일 객체 생성
    token = delim.split(sent) #정규표현식으로 sent 텍스트 분할
    print(token)
    if token[-1] == '' : token = token[:-1] #맨 뒤 빈 문자열 제거
    print(token) 
    
    ###### 실행결과
    ['Hello', 'Python', 'Coding', 'Programming', 'Study', ''] 
    ['Hello', 'Python', 'Coding', 'Programming', 'Study']
    ```

    1. r ' 문자열 형식 '

    - r = raw (그대로 사용을 의미)

    2. \[-\s.,!@;?\]+
    - \+
        - []안의 여러 문자
    - \s
        - 공백 문자 의미

3. NLTK (공개 라이브러리)
    <details>
    <summary>코드</summary>

    ```python
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tokenize import TreebankWordTokenizer
    nltk.download('punkt') #구두점 데이터 다운로드
    
    sent = "Hi! Python. Coding,   isn't, Study.."
    
    token1 = word_tokenize(sent)
    token2 = TreebankWordTokenizer().tokenize(sent)
    
    print(token1)
    print(token2)
    
    ###### 실행결과
    [nltk_data] Downloading package punkt to /home/runner/nltk_data...
    [nltk_data] Unzipping tokenizers/punkt.zip. 
    ['Hi', '!', 'Python', '.', 'Coding', ',', 'is', "n't", ',', 'Study', '..']
    ['Hi', '!', 'Python.', 'Coding', ',', 'is', "n't", ',', 'Study..']
    ```
    </details>


4. 사용자 정의 함수

    <details><summary>코드</summary>
    
    ```python
    #txtutils.py
    def simple_tokenize(txt):
    '''
    간단한 토크나이징 함수
    :param txt: 문자열
    :return: 토큰 리스트
    '''
    txt = txt.lower()
    txt = txt.replace('.', ' .')
    token = txt.split()
    return token
    ```
    </details>
    <details>
    <summary>활용</summary>
    
    ```python
    from util import txtutils as tu #util 패키지에서 txtutils를 포함하여 tu로 사용한다.

    doc = 'simple tokenize function test.'
    token = tu.simple_tokenize(doc)
    print(token)

    #출력
    ['simple', 'tokenize', 'function', 'test', '.']
    ```
    </details>

5. 실습 내용
    ```python
    #tokenization_exec.py

    import nltk
    from nltk.tokenize import TreebankWordTokenizer
    nltk.download('punkt') #구두점 데이터 다운로드

    docs = []
    docs.append("I am going to go to the store.")
    docs.append("The Science of today is the technology of tommorow.")
    docs.append("You are using pip version 3.")
    docs.append("Could not install packages due to an Error.")

    # docs list 안에는 4개의 문장이 들어있다.
    # 각 문장을 토큰화하여 실행결과에 보이는 것과 같이 [문장별 토큰 리스트]를 가진 리스트에 넣고 출력하는
    # 코드를 작성해볼 것

    ###### 실행결과
    [nltk_data] Downloading package punkt to /home/runner/nltk_data... 
    [nltk_data] Package punkt is already up-to-date! 
    [['i', 'am', 'going', 'to', 'go', 'to', 'the', 'store', '.'],
    ['the', 'science', 'of', 'today', 'is', 'the', 'technology', 'of', 'tommorow', '.'],
    ['you', 'are', 'using', 'pip', 'version', '3', '.'],
    ['could', 'not', 'install', 'packages', 'due', 'to', 'an', 'error', '.']]
    ```
<div style="text-align: right">

[목차](#목차)
</div>
---

## 3. 문장 토큰화
1. ### 출력
    <details>
    <summary>위의 실습을 출력</summary>

    ```python
    tk = TreebankWordTokenizer()
    tokens = []
    for doc in docs:
        token = tk.tokenize(doc.lower())
        tokens.append(token)
    print(tokens)
    ```
    </details>

2.  문장 토큰화
    <details>
    <summary>NLTK 이용 코드</summary>

    ```python
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import word_tokenize
    ...
    text = '\n'.join(docs)
    sentences = sent_tokenize(text) #문장 토큰 리스트
    print(sentences)
    tokens = [word_tokenize(t) for t in sentences]
    print(tokens)

    ## 결과

    #sentence tokenize
    ['I am going to go to the store.', 
    'The Science of today is the technology of tommorow.',
    'You are using pip version 3.', 
    'Could not install packages due to an Error.']

    #word tokenize
    [['I', 'am', 'going', 'to', 'go', 'to', 'the', 'store', '.'],
    ['The', 'Science', 'of', 'today', 'is', 'the', 'technology', 'of', 'tommorow', '.'],
    ['You', 'are', 'using', 'pip', 'version', '3', '.'], 
    ['Could', 'not', 'install', 'packages', 'due', 'to', 'an', 'Error', '.']]
    ```
    </details>
<div style="text-align: right">

[목차](#목차)
</div>
---

## 4. 단어 임베딩
<img src="https://ivy-hospital-413.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F53b94b0e-e2d7-46d2-86c2-fe60e1f39f80%2F%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-07-29_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.23.19.png?id=ce63509f-987b-4a65-a4b6-118ff6d81c0e&table=block&spaceId=4df7c095-f16c-49b1-9f2e-58f72fb09349&width=1230&userId=&cache=v2"></img>

1. ### 사전 구축
    1. 직접 구축
        <details>
        <summary>코드</summary>

        ```python
        import nltk
        from nltk.tokenize import word_tokenize
        nltk.download('punkt')

        vocab = {}          # python dictionary
        doc_tokens = []     # python list
        for doc in docs:
            tokens = word_tokenize(doc.lower()) #문서 토큰화 (구두점 제거 -> details로 변경)
            for word in tokens:
                if word not in vocab: #사전에 없는 단어일 경우 
                    vocab[word] = 0   #사전에 추가
                vocab[word] += 1    
            doc_tokens.append(tokens)
        print(vocab)
        print(doc_tokens)

        ### 결과
        ###----vocab data----
        {'to': 6, 'do': 8, 'is': 2, 'be': 8, '.': 6, 'or': 1, 'not': 1, 'i': 4, 'am': 3, 'what': 1, 'think': 1, 'therefore': 1, ',': 3, 'da': 3, 'let': 2, 'it': 2}
        ```
        <details>
        <summary>구두점 제거</summary>
        
        ```python
        for doc in docs:
        delim = re.compile(r'[\s,.]+') #공백문자, ',', '.'으로 구분
        tokens = delim.split(doc.lower()) #정규표현식으로 sent 텍스트 분할
        if tokens[-1] == '' :   tokens = tokens[:-1] #맨 뒤에 빈 문자열 제거
        ```
        </details>
        </details>
    
    2. 카운터 활용
        <details>
        <summary>코드</summary>

        ```python
        import re
        from collections import Counter

        doc_tokens = []     # python list
        for doc in docs:
            delim = re.compile(r'[\s,.]+')
            tokens = delim.split(doc.lower()) #정규표현식으로 sent 텍스트 분할
            if tokens[-1] == '' :   
                tokens = tokens[:-1] 
            doc_tokens.append(tokens)

        vocab = Counter(sum(doc_tokens, []))
        print(type(vocab)) #collection.Counter 타입
        vocab #딕셔너리와 비슷함
        ```
        </details>

    3. NLTK FreqDist 활용
        <details>
        <summary>코드</summary>

        ```python
        import re
        from nltk import FreqDist
        import numpy as np

        doc_tokens = []     # python list
        for doc in docs:
            delim = re.compile(r'[\s,.]+')
            tokens = delim.split(doc.lower()) #정규표현식으로 sent 텍스트 분할
            if tokens[-1] == '' :   tokens = tokens[:-1] 
            doc_tokens.append(tokens)

        vocab = FreqDist(np.hstack(doc_tokens)) #nltk.probability.FreqDist 타입
        print(type(vocab))
        print(vocab['to'])
        vocab #위와 동일한 형태 
        ```
        </details>
    
    인덱스 부여
    - 단어로 인덱스 찾기
        <details><summary>코드</summary>
        
        ```python
        word_to_id = {word[0] : id for id, word in enumerate(vocab)}
        word_to_id
        ```
        </details>
    - 인덱스로 단어 찾기
        <details><summary>코드</summary>
        
        ```python
        id_to_word = {id : word[0] for id, word in enumerate(vocab)}
        id_to_word
        ```
        </details>
    OOV(Out Of Vocabulary)
    - 빠진 단어 목록의 추가(빈도 수가 낮거나 불용어)
    - 전체의 이름을 'OOV'로 word_to_id에 추가 가능
    - <details>
        <summary>코드 예</summary>

        ```python 
        [in ] word_to_id['oov'] = len(word_to_id)+1
        [in ] word_to_id['oov']
        [out] 14
        ```
        </details>

     #### 사전 임베딩 정리
    1. 입력 문서별 토큰 리스트 작성(doc_tokens)
    2. 단어와 빈도수로 이루어진 사전 구축(vocab)
    3. 단어 : 인덱스, 인덱스 : 단어로 이루어진 사전(word_to_id, id_to_word) 구축

    <details>
    <summary>사전임베딩 코드 전문</summary>
    
    ```python
    from nltk import FreqDist
    import numpy as np
    import re

    def buildDict(docs):
        doc_tokens = []     # python list
        for doc in docs:
            delim = re.compile(r'[\s,.]+')
            tokens = delim.split(doc.lower()) 
            if tokens[-1] == '' :   tokens = tokens[:-1] 
            doc_tokens.append(tokens)

        vocab = FreqDist(np.hstack(doc_tokens))
        vocab = vocab.most_common()
        word_to_id = {word[0] : id for id, word in enumerate(vocab)}
        id_to_word = {id : word[0] for id, word in enumerate(vocab)}
        return doc_tokens, vocab, word_to_id, id_to_word    

    docs = []
    docs.append('To do is to be. To be is to do.')
    docs.append('To be or not to be. I am what I am.')
    docs.append('I think therefore I am. Do be do be do.')
    docs.append('Do do do, da da da, Let it be, let it be.')

    doc_tokens, vocab, word_to_id, id_to_word = buildDict(docs)

    print(doc_tokens)
    print(vocab)
    print(word_to_id)
    print(id_to_word)
    ```
    </details>
<div style="text-align: right">

[목차](#목차)
</div>
---

## 5. 원핫벡터(one-hot-vector)
- 1차원 배열의 저장 형태로써 배열 내 원소 중 정답을 뜻하는 원소 하나만 1이고 나머지 모든 원소는 0인 배열을 의미
    <details><summary>코드</summary>

        ```python
        import pandas as pd

        sent = 'Thomas Jefferson began building Monticello at the age of 26.'
        token = sent.split() #간단한 단어 분리
        print(token)

        ##출력결과##
        ['Thomas', 'Jefferson', 'began', 'building', 'Monticello', 'at', 'the', 'age',  'of', '26.']

        one_hot_vectors = [] #토큰별 원-핫 벡터의 리스트

        for idx, word in enumerate(token): #토큰마다 원-핫 벡터 생성 
            vector = [0 for _ in token] #모든 값이 0인 사전크기 리스트 생성
            vector[token.index(word)] = 1 #사전에서 단어가 출현한 위치만 1로 설정 
            one_hot_vectors.append(vector) #만들어진 벡터를 원-핫 벡터에 저장 

        #결과출력
        df = pd.DataFrame(one_hot_vectors, columns=vocab)
        print(df)
        ```
        결과 사진
        <img src = "https://ivy-hospital-413.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F8b9cb14a-41e4-4ba2-8173-565413024cc1%2FLectureCode_-_Replit.png?id=82d8f633-66a6-4dbf-9bea-7525efb6eca8&table=block&spaceId=4df7c095-f16c-49b1-9f2e-58f72fb09349&width=1230&userId=&cache=v2">
        </details>

- 원-핫 벡터 만들기
    1. buildDict()으로 단어사전 생성
        - 생성되는 것
            - 문서의 토큰 리스트
            - 단어 Dict
            - 단어 - 인덱스 Dict
            - 인덱스 - 단어 Dict
    2. 단어사전 토큰 리스트 요소 마다
        - 사전 크기만큼 0이 들어간 원핫 벡터 리스트 생성
        - 요소 위치의 0의 자리에 1을 대신 삽입
        - 한줄씩 벡터 추가
- 원-핫 벡터에서 원문 복구
    - 벡터마다 1의 위치 검색
    - 위치를 이용해 id-to-word으로 단어로 변경
    - 전부 찾아서 문장으로 반환

<div style="text-align: right">

[목차](#목차)
</div>
---

## 6. 이진 벡터 
- 문서당 이진 벡터 1개 
- 사전에 있는 단어만을 1로 표시
- 문서에 단어가 출현 했는지만을 확인하는 용도

### 역색인
- 이미 색인된 단어(키워드)를 이용해 단어가 포함된 문서를 찾을 수 있도록 구축해놓은 색인 자료구조
1. 원문 토큰화
2. 텍스트 전처리(불용어 제거, 어간 추출, 품사 태깅 등)
3. 색인어 추출 및 정렬
4. 색인어 당 문서 벡터 생성
5. 색인어에 해당하는 문서 벡터내 문서 랭킹(검색순위 결정)
6. 해시테이블, BTree 등의 자료구조로 역 인덱스 구축
---

## 7. TF-IDF 벡터
- TF : 용어 빈도수(Term Frequency)
    - 문서(d)에 나타난 단어(t)의 빈도수
    - 위 수의 log값 + 1 = TF, 단어가 없으면 이면 0
    - 의미 : 여러번 나오는 단어는 중요함 (문서별 계산)
- IDF : 역문헌 빈도수(Inverse Document Frequency)
    - Zipf의 법칙
        - k번째 많은 단어의 빈도수는 가장 많이 출현한 단어 빈도수의 1/k에 근접
        - 즉 최대 1000회 출현 단어가 존재 시, 4번째 많이 출현하는 단어는 250에 근접
    - log (전체 문헌 수 / 특정 단어가 출현한 문헌 수) = IDF
    - 의미 : 많은 문서에서 나올 경우 중요하지 않음(여러 문서집합으로 한번 계산)
- TF - IDF : TF × IDF
- 활용 
    - 키워드 추출
    - 순위 결정
    - 유사도 측정
- 벡터 정규화
    - 큰 문서일 수록 가중치가 올라감
    - 문서 크기 표준화 필요
    - L1 정규화, L2정규화
        - L1 : 가중치 값 / 모든 가중치의 합
        - L2 : 가중치 값 / 모든 가중치의 제곱 합의 제곱근
    - 코사인 유사도
        - A · B = ∥A∥ ∥B∥ cosθ
        - cosθ = (A · B)/(∥A∥ ∥B∥) = √Σ(A×B) / (√ΣA² · √ΣB²)
- 코사인 유사도
    - 특성
    - 문서의 정규화 효과
    - 다차원 양수공간 벡터 유사도 계산
<div style="text-align: right">

[목차](#목차)
</div>

---
## 8. 사이킷런 활용
[scikit_learn.org](https://scikit-learn.org/stable/)
1. 원핫 벡터
    1. LabelEncoder의 fit_transform 함수를 사용해 사전의 단어들을 레이블링
    2. LabelEncoder의 transform 함수를 활용해 우리가 가진 doc_tokens의 단어를 레이블링 된 숫자 벡터로 변환
    3. 1의 과정에서 만들어진 레이블링 데이터를 2차원으로 변환하여 OneHotEncoder의 fit_transform 함수를 사용해 레이블링
    4. 2의 과정에서 만들어진 벡터를 2차원으로 변환하여 OneHotEncoder의 transform 함수를 사용해 원-핫 벡터로 변환
    <details><summary>코드</summary>
    
    ```python
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    encoder = LabelEncoder()
    labels = encoder.fit_transform([word for word, id in vocab])
    for label in labels:
        print('[{:2d} : {}]'.format(label, encoder.classes_[label]))

    encode_data = np.array([encoder.transform(doc_token) for doc_token in doc_tokens])
            
    from sklearn.preprocessing import OneHotEncoder

    oh_encoder = OneHotEncoder(categories='auto')
    labels = labels.reshape(-1, 1)
    oh_labels = oh_encoder.fit_transform(labels)

    from sklearn.preprocessing import OneHotEncoder

    oh_encoder = OneHotEncoder(categories='auto')
    labels = labels.reshape(-1, 1)
    oh_labels = oh_encoder.fit_transform(labels)

    #출력 
    from sklearn.preprocessing import OneHotEncoder

    oh_encoder = OneHotEncoder(categories='auto')
    labels = labels.reshape(-1, 1)
    oh_labels = oh_encoder.fit_transform(labels)
    ```
    
    </details>
2. 카운터 벡터
    <details><summary>코드</summary>
    
    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    docs = []
    docs.append('To do is to be. To be is to do.')
    docs.append('To be or not to be. II am what II am')
    docs.append('II think therefore II am. Do be do be do.')
    docs.append('Do do do da da da. Let it be let it be.')

    cnt_vectr = CountVectorizer()
    vectors = cnt_vectr.fit_transform(docs)

    print(cnt_vectr.vocabulary_)
    print(cnt_vectr.get_feature_names())
    print(vectors.toarray())
    print(pd.DataFrame(vectors.toarray(),
                    columns=cnt_vectr.get_feature_names()))
    ```
    </details>
3. TF-IDF 벡터
    <details><summary>코드</summary>
    
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd

    tfidf = TfidfVectorizer().fit(docs)
    dtm = tfidf.transform(docs).toarray()

    df = pd.DataFrame(dtm, columns=tfidf.get_feature_names())
    print(df)
    print(sorted(tfidf.vocabulary_.items()))
    ```
    </details>
- TF 계산 : 빈도수 카운트. 우리가 수업에서 만든 코드는 로그를 취함   
- IDF 계산 : 스무딩을 수행할 경우, 전체 문서수(N)+1, 단어출현 문서수(DF)+1을 해서 로그값을 취하며, 그 결과에 1을 더함으로 최종 IDF 값을 취함
- L2 정규화를 진행함 → 벡터 요소의 제곱합이 1이 되도록 정규화 → 벡터 유사도 계산 시 벡터의 크기를 정규화 해야 한다.
- 로그 : 수업에서는 밑이 2인 로그를 취했으나, 패키지에서는 np.log()를 취함. np.log()는 자연로그임. 

<div style="text-align: right">

[목차](#목차)
</div>

---
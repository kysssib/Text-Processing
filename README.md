# **Text-Processing**

## **목차**
- <details><summary>중간</summary>

    1. [텍스트 처리의 단계](#1-텍스트-처리의-단계)
    2. [토큰화](#2-토큰화)
    3. [문장 토큰화](#3-문장-토큰화)
    4. [단어 임베딩](#4-단어-임베딩)
    5. [원핫 벡터](#5-원핫벡터one-hot-vector)
    6. [이진벡터](#6-이진-벡터)
    7. [TF-IDF벡터/코사인 유사도/정규화](#7-tf-idf-벡터)
    8. [사이킷런](#8-사이킷런-활용)
    </details>

- <details><summary>기말</summary>

    9. [정규표현식](#9-정규표현식regular-expression)
    10. [형태소 분석](#10-형태소-분석)
    11. [동시발생행렬](#11-동시발생행렬)
    12. [CBOW](#12-cbow)
    13. [TMDB 실습](#13-tmdb-실습)
    14. [네이버 영화 실습](#14-네이버-영화-실습)
    15. [사전 임베딩 자료 실습](#15-사전-임베딩-자료-실습)
    </details>

---

## 1. 텍스트 처리의 단계
<details>

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
</details>
<div style="text-align: right">

[목차](#목차)
</div>

---

## 2. 토큰화

<details>

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

</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 3. 문장 토큰화

<details>

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

</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 4. 단어 임베딩

<details>

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

</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 5. 원핫벡터(one-hot-vector)

<details>

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

- 형태 
    - 가로 : 단어 사전
    - 세로 : 단어 순서

    ||W1|W2|W3|. . .|Wn|
    |--|--|--|--|--|--|
    |I1|1|0|0|. . .|0|
    |I2|0|0|1|. . .|0|
    |I3|0|1|0|. . .|0|
    |.<br>.<br>.|.<br>.<br>.|.<br>.<br>.|.<br>.<br>.|.<br>&nbsp;&nbsp;&nbsp;.<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.|.<br>.<br>.|
    |Ix|0|0|0|. . .|1|

    예시는 W1 W3 W2 ... Wn 을 나타냄

</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 6. 이진 벡터 

<details>

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


</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 7. TF-IDF 벡터

<details>

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
    - 형태

        ||T1|T2|T3|. . .|Tn|
        |--|--|--|--|--|--|
        |D1|w(1-1)|w(1-2)|w(1-3)|. . .|w(1-n)|
        |D2|w(2-1)|w(2-2)|w(2-3)|. . .|w(2-n)|
        |D3|w(3-1)|w(3-2)|w(3-3)|. . .|w(3-n)|
        |.<br>.<br>.|.<br>.<br>.|.<br>.<br>.|.<br>.<br>.|.<br>&nbsp;&nbsp;&nbsp;.<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.|.<br>.<br>.|
        |Dn|w(n-1)|w(n-2)|w(n-3)|. . .|w(n-n)|

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


</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 8. 사이킷런 활용

<details>

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

</details>

<div style="text-align: right">

[목차](#목차)
</div>

---
---

## 9. 정규표현식(Regular Expression)

<details>

- 특정 문자의 집합 또는 문자열을 기호식으로 바꾸어 놓는 방법
    - 문자열 집합을 정확하게 표현하기 위해 사용하는 규칙

- 사용처
    1. 찾고자 하는 문자열 형식 지정(정규 표현식 생성)
    2. 텍스트 데이터에서 정규표현식 해당 문자열 검색(문자열 검색)
    3. 찾아서 작업 수행
        1. 제대로 작성 하였는가
        2. 다른 문자열로 치환
        3. 몇번 나왔는가 검사 등
    
- [정규 표현식 링크](https://regex101.com/)

### 정규표현식

1. [일치하는 문자열](#일치하는-문자열)
2. [.(dot)](#dot)
3. [\[ \](괄호)](#괄호)
4. [\\(메타문자)](#메타문자)
5. [\+(하나 이상 문자)](#하나-이상-문자)
6. [\*(없거나 하나 이상 문자)](#없거나-하나-이상-문자)
7. [?(없거나 하나인 문자)](#없거나-하나인-문자)
8. [\{\}(특정 횟수 만큼 출현)](#특정-횟수-만큼-출현)
9. [\(\)(하위 표현식)](#하위-표현식)
10. [논리연산자](#논리연산자)
11. [탐욕적 vs 게으른 수량자](#탐욕적-vs-게으른-수량자)
12. [문자열 시작, 끝](#문자열-시작-끝)

#### 일치하는 문자열
express : is
- 가장 먼저 나오는 is를 확인

#### \.(dot)
express : i.
- 문자 i와 모든 문자 1개를 확인 iㅁ

#### \[\](괄호)
express : \[Ee\]\[Rr\]
- ER Er eR er 모두 가능 즉 괄호 안 중 문자 1개로 인식
    - \[A-Z\]로 범위 지정도 가능
    - \[A-Za-Z0-9\] : 모든 알파벳이나 숫자 중 1문자
    - \[^0-9\] : 숫자가 아닌 문자

#### \\(메타문자)
express : \\\[\[0-9\]\\\]
- \[와 0-9 중 한 문자와 \]
    - \[\]를 그 문자 그대로 사용하게 함

    <details>
        <summary>설명표</summary>

    메타문자

    |메타문자|설명|
    |---|---|
    |\b|백스페이스|
    |\f|페이지 넘김|
    |\n|줄바꿈|
    |\r|엔터|
    |\t|탭|
    |\v|수직탭|

    문자 클래스

    |클래스|메타문자|설명|일반정규표현식|
    |--|--|--|--|
    |숫자|\d|숫자 하나|\[0-9\]|
    ||\D|숫자 제외 문자 하나|\[^0-9\]|
    |문자|\w|대문자, 소문자, 숫자, 밑줄 문자 중 하나|\[A-Za-z0-9_\]|
    ||\W|대문자, 소문자, 숫자, 밑줄 문자 제외 문자 하나|\[^A-Za-z0-9_\]|
    |공백|\s|모든 공백 문자 중 하나|\[\n\r\t\v\f\]|
    ||\S|모든 공백 문자 제외 문자 하나|\[^\n\r\t\v\f\]|


    [추가적인 정보](https://developer.mozilla.org/ko/docs/Web/JavaScript/Guide/Regular_expressions)
    </details>

#### \+(하나 이상 문자)
express : \w+
- 하나 이상의 대문자, 소문자, 숫자, 밑줄 문자가 출현
    <details>
    <summary>코드</summary>

    ```python
    import re
    doc = ["send : host@server.com",
    "recv : guest@client.com",
    "return : admin_1@test.server.com",
    "fwd : admin.programming@test.server.co.kr"]
    text = "\n".join(doc)

    pattern = re.compile(r"[\w.]+@[\w.]+\.\w+", re.MULTILINE)

    print(pattern.findall(text))
    result = pattern.finditer(text)
    for m in result:
    print(m.group(), m.span(), m.start(), m.end()))
    ```
    </details>

#### \*(없거나 하나 이상 문자)
express : 0\w*
- 0 뒤에 없거나 하나 이상의 대문자, 소문자, 숫자, 밑줄 문자가 출현

#### ?(없거나 하나인 문자)
express : 0\w?
- 0 뒤에 없거나 하나의 대문자, 소문자, 숫자, 밑줄 문자가 출현

#### \{\}(특정 횟수 만큼 출현)
express : \w\{6\}
- 대문자, 소문자, 숫자, 밑줄 문자가 6번 출현
- 최대, 최소 지정 가능 \{최소, 최대\} 최대 반복 생략 가능

#### \(\)(하위 표현식)
express : \(&nbsp\)\{3\}
- 공백이 3개

#### 논리연산자
express : \(\[0-9\]\)|\(\[A-Z\]\)
- 숫자 1개 또는 대문자 1개
- | : or , & : and

#### 탐욕적 vs 게으른 수량자
express : \<pP\>.+\</pP\>
- \<pP\>와 하나 이상의 모든 문자와 \</pP\>
- 단 + 수량자는 탐욕적으로 검출 시 뒤에서부터 작용
- 따라서 결과값이 result = \<p\>sample text\</p\>\<p\>text 1\</p\>\<P\>text 2\</P\>으로 생성됨

express : \<pP\>.+?\</pP\>
- \<pP\>와 하나 이상의 모든 문자와 \</pP\>
- 위와 동일해 보이지만 +에 ?를 붙여 게으른 수량자로 사용
- 앞에서부터 최소한으로 적게
- \<p\>sample text\</p\>\<p\>text 1\</p\>\<P\>text 2\</P\> 검출 시 
    - \<p\>sample text\</p\>
    - \<p\>text 1\</p\>
    - \<P\>text 2\</P\>
- 으로 생성됨

#### 문자열 시작, 끝
express : ^\(0.+9\)$
- 시작이 0 하나 이상의 문자와 끝이 9
- 시작 끝 패턴을 명확히 지정할때 사용

</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 10. 형태소 분석

<details>

- 형태소 : 언어에서 의미를 가지는 가장 작은 단위

    1. 토큰화 시 형태소 분석을 할 것인지 결정
    2. 사용할 경우 목적에 따라 사용 범위 결정
    3. 가장 핵심 기능 : 품사 태깅(part of tagging POS tagging)

- 품사
    1. 체언 : 명사, 대명사, 수사...
    2. 용언 : 동사, 형용사...
    3. 독립언 : 부사, 감탄사...
    4. 기능어 : 조사, 어미, 접사...

- 한글의 형태소 분석
    1. 다양한 조사 활용
    2. 불규칙 형 변환
    3. 언어 파괴적 요소(띄어쓰기, 오타)
    - 자유자재의 언어 구사, 규칙 파괴에도 이해 가능
    - 이러한 요소가 검색엔진, 자연어 처리 시스템 구현에 큰 한계

- 형태소 분석의 목적 = 색인작업

- Konlpy 사용

     ```python
    !curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash
    ```

    - 사용 예시
        ```python
        from konlpy.tag import Okt, Mecab
        okt = Okt()
        okt.morphs('형태소 분석할 Text 입력')
        ```


    <details> 
    <summary>Konlpy 안의 형태소</summary>

    - Hannaum : KASIT
    - Kkma : 서울대
    - Komoran : Shineware
    - Mecab : 일본어용을 한국어 사용가능하게 수정
    - Open Korean Text : 과거 트위터 형태소 분석기
    </details>

- TF/IDF 적용
    - <details><summary>TF/IDF 벡터 생성</summary>

        ```python
        docs = doc.split('\n')
        r = []
        for line in docs:
            token = mecab.morphs(line)
            txt = " ".join(token)
            r.append(txt)

        from sklearn.feature_extraction.text import TfidfVectorizer
        #tfidf 벡터 메트릭스 생성
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(r)
        print('type of tfidf_matrix {}'.format(type(tfidf_matrix)))
        print('shape of tfidf_matrix {}'.format(tfidf_matrix.shape))
        
        ##출력결과
        type of tfidf_matrix <class 'scipy.sparse.csr.csr_matrix'>
        shape of tfidf_matrix (357, 1081)
        ```
        </details>

    - <details><summary>각 문서별 TF/IDF 값이 높은 단어 추출</summary>

        ```python
        tfidf_table = tfidf_matrix.toarray()
        keywords = []

        for weight in tfidf_table:
            w_vec = list(enumerate(weight))
            w_vec = sorted(w_vec, key=lambda x : x[1], reverse=True)
            print(w_vec[:3])
            keywords.append(w_vec)

        #출력결과
        [(361, 0.8233074079998407), (1049, 0.5675957293114385), (0, 0.0)]
        [(0, 0.0), (1, 0.0), (2, 0.0)]
        [(664, 0.297683329858021), (412, 0.2253201732864864), (483, 0.19845555323868067)]
        [(0, 0.0), (1, 0.0), (2, 0.0)]
        [(949, 1.0), (0, 0.0), (1, 0.0)]
        [(215, 0.6937741027461318), (412, 0.5251261504196152), (361, 0.492869171793363)]
        ```
        </details>

    - <details><summary>전체 문서 중 가장 TF/IDF가 높은 문서 추출</summary>

        ```python
        import numpy as np
        def tfidf_rank(tfidf_matrix):
            rank = []
            avg, stddev = 0.0, 0.0
            
            #문서 별 tfidf 가중치의 합 계산 : (문서id, 가중치 합)
            for idx, tfidf in enumerate(tfidf_matrix):
                rank.append((idx, tfidf.sum()))

            #가중치의 합이 높은 문서 순으로 정렬
            rank.sort(key=lambda x : x[1], reverse=True)

            #tfidf의 평균과 표준편차 계산
            tfidf_sum = [tfidf.sum() for tfidf in tfidf_matrix]
            avg = np.mean(tfidf_sum)
            stddev = np.std(tfidf_sum)
            return rank, avg, stddev
            
        rank, avg, stddev = tfidf_rank(tfidf_matrix)

        print(rank[:2])
        print('avg = {}, stddev = {}'.format(avg, stddev))

        #랭크가 높은 문서 5개의 원문을 추출하여 rank_doc에 저장 후 출력
        rank_doc = [docs[doc_id[0]] for doc_id in rank[:5]]
        rank_doc

        #출력결과
        [(2, 8.937515671877552), (181, 5.81765311803503)]
        avg = 2.949446365690751, stddev = 1.043877787780237
        
        ['유구한 역사와 전통에 빛나는 우리 대한국민은 3·1운동으로 건립된...'
        ' 제76조 ① 대통령은 내우·외환·천재·지변 또는 중대한 재정·경제상의...'
        '③체포·구속·압수 또는 수색을 할 때에는 적법한 절차에 따라 검사의...'
        '⑦피고인의 자백이 고문·폭행·협박·구속의 부당한 장기화 또는 기망 기타의...'
        ' 제65조 ① 대통령·국무총리·국무위원·행정각부의 장·헌법재판소...'
        ```
        </details>
    

</details>

<div style="text-align: right">

[목차](#목차)
</div>

---


## 11. 동시발생행렬

<details>

- 단어의 분산 표현
    - 통계기반
        - 단어 출현 획구에 기반한 처리
        - 의미 파악 불가
        - 희소벡터(TF/IDF)
    - 추론기반
        - 주변 단어와 관계에 기반한 처리
        - 의미 추론 가능
        - 밀집벡터

- 맥락
    - 정의 : 사물 따위가 서로 이어져 있는 관계나 연관성
    - all the facts, opinions, etc. relating to a particular thing or event

    - 표현
        -  동시발생행렬
        - 특정 단어의 주변에 나타나는 단어의 횟수를 기록하는 간단한 방법
    
    - 예문 : **you say goodbye and i say hello.**
        - ||you|say|goodbye|and|i|hello|.|
            |--|--|--|--|--|--|--|--|
            |you|0|1|0|0|0|0|0|
            |say|1|0|1|0|1|1|0|      
            |goodbye|0|1|0|1|0|0|0|     
            |and|0|0|1|0|1|0|0|     
            |i|0|1|0|1|0|0|0|    
            |hello|0|1|0|0|0|0|1|        
            |.|0|0|0|0|0|1|0|   
        
        <details><summary>동시발생행렬 코드</summary>
        
        ```python
        def create_co_matrix(corpus, vocab_size, window_size=1):
            '''동시발생 행렬 생성
            :param corpus: 말뭉치(단어 ID 목록)
            :param vocab_size: 어휘 수
            :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
            :return: 동시발생 행렬
            '''
            corpus_size = len(corpus)
            co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

            for idx, word_id in enumerate(corpus):
                for i in range(1, window_size + 1):
                    left_idx = idx - i
                    right_idx = idx + i
                    
                    #기준 단어의 왼쪽 값 추가
                    if left_idx >= 0:
                        left_word_id = corpus[left_idx]
                        co_matrix[word_id, left_word_id] += 1

                    #기준 단어에 오른쪽 값 추가
                    if right_idx < corpus_size:
                        right_word_id = corpus[right_idx]
                        co_matrix[word_id, right_word_id] += 1

            return co_matrix

        co_matrix = create_co_matrix(corpus, len(corpus))
        co_matrix

        #출력결과
        array([[0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
        ```
        </details>

        <details><summary>동시발생행렬 활용 코드</summary>
        
        코사인 유사도
            
        ```python
        import pandas as pd

        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import linear_kernel

        cos_sim = cosine_similarity(co_matrix, co_matrix)
        pd.DataFrame(cos_sim, columns=word_to_id.keys())

        #출력결과
                    you	     say	 goodbye	     and	       i	   hello	  .
        0	1.000000	0.000000	0.707107	0.000000	0.707107	0.707107	0.0
        1	0.000000	1.000000	0.000000	0.707107	0.000000	0.000000	0.5
        2	0.707107	0.000000	1.000000	0.000000	1.000000	0.500000	0.0
        3	0.000000	0.707107	0.000000	1.000000	0.000000	0.000000	0.0
        4	0.707107	0.000000	1.000000	0.000000	1.000000	0.500000	0.0
        5	0.707107	0.000000	0.500000	0.000000	0.500000	1.000000	0.0
        6	0.000000	0.500000	0.000000	0.000000	0.000000	0.000000	1.0
        ```
            
        입력 쿼리와 유사도가 높은 단어 반환
            
        ```python
        def most_similar(query, word_to_id, id_to_word, word_matrix, top=3):

            if query not in word_to_id:
                print('{}를 찾을 수 없음.'.format(query))
                return
                
            word_vector = np.array(word_matrix[word_to_id[query]])
            word_vector = word_vector.reshape(1, -1)

            sim = cosine_similarity(word_vector, word_matrix)
            sim = sim[0]
            sim = [(id, cos) for id, cos in enumerate(sim)]
            sim = sorted(sim, key=lambda x: x[1], reverse=True)

            return sim[1:top+1]

        rank = most_similar('you', word_to_id, id_to_word, co_matrix)
        for r in rank:
            print(id_to_word[r[0]], r[1])
        ```
            
        </details>
- 점별 상호 정보량 처리
     - $PMI(x,y) = log_2\frac{P(x,y)}{P(x)P(y)}=log_2\frac{\frac{n(x,y)}{N}}{\frac{n_x}{N}\frac{n_y}{N}} = log_2\frac{n(x,y)N}{n_xn_y}$
     - 전체 동시출현 횟수가 10,000회라고 할 때, 다음과 같은 발생 횟수를 나타낸다고 가정할 때
        1. the : 1,000회
        2. car : 20회
        3. drive : 10회
        4. the, car 동시발생 : 10회
        5. car, drive 동시발생 : 5회

        - $PMI(the,car) = log_2\frac{10\cdot10000}{1000\cdot20}=log_2\frac{10}{2}\approx2.32$
        - $PMI(car,drive) = log_2\frac{5\cdot10000}{20\cdot10}=log_2 250\approx7.97$
            - the같은 고빈도 단어 출현 시 분모값 증가, 전체값 감소
            - 출현 빈도 없을 시 $log_20=-\infty$가 되므로 양의 값만 가지는 함수 필요
            - $PPMI(x,y) = max(0,PMI(x,y))$
            - <details><summary>코드</summary>
            
                ```python
                def ppmi(C, verbose=False, eps = 1e-8):
                    '''PPMI(점별 상호정보량) 생성
                    :param C: 동시발생 행렬
                    :param verbose: 진행 상황을 출력할지 여부
                    :return:
                    '''
                    M = np.zeros_like(C, dtype=np.float32)
                    N = np.sum(C)
                    S = np.sum(C, axis=0)
                        total = C.shape[0]*C.shape[1]
                        cnt = 0
                    print('N = {}, S = {}'.format(N, S))

                    for i in range(C.shape[0]):
                        for j in range(C.shape[1]):
                            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps) #PMI 계산
                            M[i, j] = max(0, pmi) #Positive

                            if verbose: #진행상황
                                cnt += 1
                                if cnt % (total//100 + 1) == 0:
                                    print('%.1f%% 완료' % (100*cnt/total))
                    return M

                W = ppmi(co_matrix)

                np.set_printoptions(precision=3)  # 유효 자릿수를 세 자리로 표시
                print('동시발생 행렬')
                print(co_matrix)
                print('-'*50)
                print('PPMI')
                print(W)

                #출력결과
                N = 14, S = [1 4 2 2 2 2 1]
                동시발생 행렬
                [[0 1 0 0 0 0 0]
                [1 0 1 0 1 1 0]
                [0 1 0 1 0 0 0]
                [0 0 1 0 1 0 0]
                [0 1 0 1 0 0 0]
                [0 1 0 0 0 0 1]
                [0 0 0 0 0 1 0]]
                --------------------------------------------------
                PPMI
                [[0.    1.807 0.    0.    0.    0.    0.   ]
                [1.807 0.    0.807 0.    0.807 0.807 0.   ]
                [0.    0.807 0.    1.807 0.    0.    0.   ]
                [0.    0.    1.807 0.    1.807 0.    0.   ]
                [0.    0.807 0.    1.807 0.    0.    0.   ]
                [0.    0.807 0.    0.    0.    0.    2.807]
                [0.    0.    0.    0.    0.    2.807 0.   ]]
                ```
                </details>

- 특잇값 분해(SVD)
    - 선형대수에서 특잇값 분해(Singular Value Decomposition)는 행렬을 분해하는 방식 중 하나
    - 행렬의 차원 감소 위한 방법으로 활용 -> 고유값 분해의 일반화 과정
        <img width="1000" alt="fig_2-8" src="https://github.com/kysssib/Text-Processing/assets/113497500/4d4478b1-e05a-4b4a-8edb-62ab2de39594">

        차원이 축소되어도 본질적 특성을 가진 값을 구별할 수 있도록 fit(적합)시켜야함
    - 사용 이유
        1. 희소벡터는 대부분 0의 값
        2. 희소벡터 -> 밀집벡터(대부분 0이 아닌 값) 효율성 up
        3. 특잇값 분해로 본질적 값에 적합하도록 차원을 줄여 근사시킴

    - 사용 방법
        - $X = USV^T$
            - U : 직교 행렬이며 원본의 행
            - S : 대각 행렬 (대각성분 외 모두 0) 특잇값 큰 순서대로 나열, U의 중요도 순서
            - V : 직교 행렬이며 원본의 열
            <img width="1000" alt="fig_2-9" src="https://github.com/kysssib/Text-Processing/assets/113497500/79bb2135-bfb1-428a-915e-c9ea13445ea8">

    - 원본 복원
        - S벡터(특잇값)의 원소 중 값이 낮은 값을 제거하고 SVD를 수행시 원복은 불가하나 근사값으로 복원이 가능
        <img width="1000" alt="스크린샷_2022-04-27_12 44 47" src="https://github.com/kysssib/Text-Processing/assets/113497500/d3644891-8caf-4ce8-82c6-4e6282571a2a">
    
    - <details><summary>동시발생행렬에 적용</summary>

        ```python
        U, S, VT = np.linalg.svd(W) #PPMI행렬인 W를 SVD화
        print("동시발생행렬\n", np.round(co_matrix, 3))
        print("PPMI적용행렬\n", np.round(W, 3))
        print("U행렬\n", np.round(U, 3))
        print("S행렬(대각요소값)\n", np.round(S, 3))

        S_matrix = np.diag(S) #대각성분 추출
        W_ = np.dot(np.dot(U, S_matrix), VT) #U와 대각성분 행렬 곱 후 VT와 행렬 곱
        print("SVD 복원 행렬 : \n",np.round(W_, 3))

        #출력결과
        동시발생행렬
        [[0 1 0 0 0 0 0]
        [1 0 1 0 1 1 0]
        [0 1 0 1 0 0 0]
        [0 0 1 0 1 0 0]
        [0 1 0 1 0 0 0]
        [0 1 0 0 0 0 1]
        [0 0 0 0 0 1 0]]
        PPMI적용행렬
        [[0.    1.807 0.    0.    0.    0.    0.   ]
        [1.807 0.    0.807 0.    0.807 0.807 0.   ]
        [0.    0.807 0.    1.807 0.    0.    0.   ]
        [0.    0.    1.807 0.    1.807 0.    0.   ]
        [0.    0.807 0.    1.807 0.    0.    0.   ]
        [0.    0.807 0.    0.    0.    0.    2.807]
        [0.    0.    0.    0.    0.    2.807 0.   ]]
        U행렬
        [[ 0.341 -0.    -0.121 -0.    -0.932 -0.    -0.   ]
        [ 0.    -0.598  0.     0.18   0.    -0.781  0.   ]
        [ 0.436 -0.    -0.509 -0.     0.225 -0.    -0.707]
        [ 0.    -0.498  0.     0.68  -0.     0.538  0.   ]
        [ 0.436 -0.    -0.509 -0.     0.225 -0.     0.707]
        [ 0.709 -0.     0.684 -0.     0.171 -0.     0.   ]
        [-0.    -0.628 -0.    -0.71   0.     0.317 -0.   ]]
        S행렬(대각요소값)
        [3.168 3.168 2.703 2.703 1.514 1.514 0.   ]
        SVD 복원 행렬 : 
        [[ 0.     1.807  0.    -0.     0.     0.     0.   ]
        [ 1.807 -0.     0.807  0.     0.807  0.807  0.   ]
        [ 0.     0.807 -0.     1.807  0.     0.     0.   ]
        [ 0.    -0.     1.807  0.     1.807 -0.     0.   ]
        [ 0.     0.807 -0.     1.807  0.     0.     0.   ]
        [ 0.     0.807  0.    -0.     0.     0.     2.807]
        [ 0.     0.     0.    -0.     0.     2.807 -0.   ]]
        ```
        </details> 

- 종합 코드

    1. <details><summary>사전 구축</summary>
    
        ```python
        #사전 구축 함수
        from nltk import FreqDist
        import numpy as np
        import re
        import nltk
        nltk.download('stopwords')

        from nltk.corpus import stopwords

        sw = stopwords.words('english')

        def buildDict(docs):
            doc_tokens = []     # python list
            for doc in docs:
                delim = re.compile(r'[\s,.]+')
                tokens = delim.split(doc.lower()) 
                tokens = [t for t in tokens if t not in sw]
                if tokens[-1] == '' :   tokens = tokens[:-1] 
                doc_tokens.append(tokens)

                
            vocab = FreqDist(np.hstack(doc_tokens))
            vocab = vocab.most_common()
            word_to_id = {word[0] : id for id, word in enumerate(vocab)}
            id_to_word = {id : word[0] for id, word in enumerate(vocab)}
            corpus = np.array([id for id, _ in enumerate(vocab)])
            return doc_tokens, corpus, word_to_id, id_to_word

        # 파일을 불러와 사전 생성

        import pandas as pd

        with open('./sample_data/sample.txt', 'r') as f:
        docs = f.readlines()

        for id, doc in enumerate(docs):
        print('[{}] : {}...'.format(id, doc[:30])) #예문 출력
        
        doc_tokens, corpus, word_to_id, id_to_word = buildDict(docs)

        #출력결과
        [0] : BTS, also known as the Bangtan...
        [1] : [5] The septet—consisting of m...
        [2] : Originally a hip hop group, th...
        [3] : Their lyrics, often focused on...
        [4] : Their work also often referenc...
        [5] : After debuting in 2013 with th...
        [6] : The group's second Korean stud...
        [7] : By 2017, BTS crossed into the ...
        [8] : They became the first Korean g...
        [9] : BTS became one of the few grou...
        [10] : In 2020, BTS became the first ...
        [11] : Their follow-up releases "Sava...
        [12] : Having sold over 20 million al...
        [13] : They are the first Asian and n...
        [14] : Featured on Time's internation...
        [15] : The group's numerous accolades...
        [16] : Outside of music, they partner...
        ```
        </details>

    2. <details><summary>동시발생행렬과 PPMI 생성</summary>

        ```python
        def create_co_matrix(corpus, vocab_size, window_size=1):
        # 동시발생 행렬 생성
        # :param corpus: 말뭉치(단어 ID 목록)
        # :param vocab_size: 어휘 수
        # :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
        # :return: 동시발생 행렬
        
        corpus_size = len(corpus)
        co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

        for idx, word_id in enumerate(corpus):
            for i in range(1, window_size + 1):
                left_idx = idx - i
                right_idx = idx + i

                if left_idx >= 0:
                    left_word_id = corpus[left_idx]
                    co_matrix[word_id, left_word_id] += 1

                if right_idx < corpus_size:
                    right_word_id = corpus[right_idx]
                    co_matrix[word_id, right_word_id] += 1

        return co_matrix

        def ppmi(C, verbose=False, eps = 1e-8):
        # PPMI(점별 상호정보량) 생성
        # :param C: 동시발생 행렬
        # :param verbose: 진행 상황을 출력할지 여부
        # :return:

        M = np.zeros_like(C, dtype=np.float32)
        N = np.sum(C)
        S = np.sum(C, axis=0)

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
                M[i, j] = max(0, pmi)
        return 
        
        window_size = 2

        vocab_size = len(word_to_id)
        print('동시발생행렬 계산')
        C = create_co_matrix(corpus, vocab_size, window_size)
        W = ppmi(C)

        print(C[0,:10])
        print(W[0, :10])

        #출력결과
        동시발생행렬 계산
        [0 1 1 0 0 0 0 0 0 0]
        [0.        7.412217  6.9971795 0.        0.        0.        0.        0.        0.        0.       ]
        
        ```
        </details>

    3. <details>
            <summary>SVD 생성</summary>
            
            
            ```python
            from sklearn.utils.extmath import randomized_svd
            wordvec_size = 100

            #행렬 분해
            U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

            from sklearn.metrics.pairwise import cosine_similarity
            def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):

                if query not in word_to_id:
                    print('{}를 찾을 수 없음.'.format(query))
                    return

                word_vector = np.array(word_matrix[word_to_id[query]]) #쿼리 단어 벡터 추출
                word_vector = word_vector.reshape(1,-1) #cosine_similarity 위해 벡터 형상 조정
                sim = cosine_similarity(word_vector, word_matrix)
                sim = sim[0] #벡터 형상 조정 ([[]] -> [])
                sim = [(id, cos) for id, cos in enumerate(sim)] #id, 유사도쌍으로  정리
                sim = sorted(sim, key=lambda x: x[1], reverse=True) #유사도 높은 순 정렬

                return sim[1:top+1]

            rank = most_similar('world', word_to_id, id_to_word, U)
            for r in rank:
            print(id_to_word[r[0]], r[1])

            #출력결과
            artist 0.5381843
            known 0.4821964
            stadium 0.44456828
            influential 0.397606
            best-selling 0.22235
            802
            ```
        </details>

    - [자세한 사항은 코드 짠거 함 봐보기](https://colab.research.google.com/drive/1rg6G8-n5Zl2JmtmENVfwbYCt7gdgBjvp?hl=ko#scrollTo=JFL1SHxtHOch)

</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 12. CBOW

<details>

- 동시발생행렬의 문제점
    - 말뭉치 증가 -> 단어벡터 커짐
    - 사전 단어 100만 -> 벡터 크기 동일
    - 특잇값 분해(SVD) $n \times n$ 행렬 -> $O(n^3)$의 시간 소요
- The machine is easy to use.
<img width="1548" alt="스크린샷 2022-03-18 15 11 43" src="https://github.com/kysssib/Text-Processing/assets/113497500/71da54df-7a9e-4f51-baba-d97722e8c3d9">
    - 원핫벡터 표현
    - 사전 크기 : 7

- 맥락과 입출력
<img width="1411" alt="스크린샷 2022-03-18 15 11 52" src="https://github.com/kysssib/Text-Processing/assets/113497500/1906c45e-02e3-4067-8ca1-b1043c3777a1">

- CBOW의 입출력
<img width="1713" alt="스크린샷 2022-03-18 15 12 00" src="https://github.com/kysssib/Text-Processing/assets/113497500/1beefb3a-5bf5-45f8-be34-e2f6664773fa">
    - 입력 : 맥락 정보
    - 출력 : 맥락에 해당하는 타겟 단어
    - 학습과정 : 맥락정보로 타겟단어를 추측 가능하게 계속 학습
    - 가중치 값을 단어 분산표현으로 사용

- 동작 과정
    <img width="1787" alt="스크린샷 2022-03-18 15 12 11" src="https://github.com/kysssib/Text-Processing/assets/113497500/ad484242-5d3c-4bf0-a898-c8f3032db9ed">
    - 입력과 출력해야하는 정보
    - 단어사전 크기 : 7

    <img width="1634" alt="스크린샷 2022-03-18 15 12 23" src="https://github.com/kysssib/Text-Processing/assets/113497500/e2233899-5dd3-47ed-8bf5-d7fb0afb3f38">

    - 입력레이어(V) : 7 * 4 행렬
        - 현재는 초기설정이라 랜덤값이 들어감
            ||$W_{in1}$|$W_{in2}$|$W_{in3}$|$W_{in4}$|
            |---|---|---|---|---|
            |the|...|...|...|...|
            |machine|0.5|0.2|0.7|1.2|
            |...|...|...|...|...|
            |easy|0.7|1.3|2.8|0.2|
            |...|...|...|...|...|

        - 히든 레이어 계산
            - $h=\frac{W_{in-macine}+W_{in-easy}}{2\times windowsize}$

    - 히든레이어(M) : 현재는 4 보통 100~300
        - $[h1,h2,h3,h4]$
        - $h1=\frac{0.5+0.7}{2\times 1}=0.6$
        - $h2=\frac{0.2+1.3}{2\times 1}=0.7$
        - $h3=\frac{0.7+2.8}{2\times 1}=1.7$
        - $h4=\frac{1.2+0.2}{2\times 1}=0.7$
        - $= [0.6, 0.7, 1.7, 0.7]$

    - 가중치계산 : 4 * 7 행렬(초기는 랜덤값 -> 학습하며 조정 -> 타겟 벡터에 근접하도록)
        - 가중치 값 계산 (여기도 초기엔 랜덤한 값)
            ||$W_{out1}$|$W_{out2}$|$W_{out3}$|$W_{out4}$|$W_{out5}$|$W_{out6}$|$W_{out7}$|
            |---|---|---|---|---|---|---|---|
            |h1|$x_{1}$|...|...|...|...|...|...|
            |h2|$x_{2}$|...|...|...|...|...|...|
            |h3|$x_{3}$|...|...|...|...|...|...|
            |h4|$x_{4}$|...|...|...|...|...|...|
        - 출력리스트 = [out1, out2, out3, out4, out5, out6, out7]
        - $score = h\times W_{out}$
        - $out1 = (h1\times W_{out1}|h1) + $<br>    $(h2\times W_{out1}|h2) + $<br>     $(h3\times W_{out1}|h3) + $<br>    $(h4\times W_{out1}|h4)$
        - $= (0.6\times x_{1}) + $<br>    $(0.7\times x_{2}) + $<br>    $(1.7\times x_{3}) + $<br>    $(0.7\times x_{4})  = 0.7$
        - 가중치 결과값 = [0.7, 1.8, 3.2, 0.9, 0.8, 0.1, 0.1]
        - 실제 타겟 값 = [0, 0, 1, 0, 0, 0, 0]
        - 각각 대조 비교
            - 스코어 정규화
            - 오차 계산
            - 가중치 값 조정(최적화)
- 가중치 갱신
    - softmax 함수
        1. score 벡터 모든 값을 0~1 사이로 
        2. 벡터 요소 총합이 1이 되도록(정규화)
        3. 요소의 값은 확률료 표현이 가능해짐

    - 손실 함수
        1. softmax 함숫값과 출력층의 오차 계산
        2. 오차 계산 바탕 가중치 갱신
    
    - 최적화
        1. softmax 함숫값과 출력층과의 오차 계산함수
        2. 확률적경사하강법(SGD), ADAM등 최적화 함수 사용

    - batch : 입력데이터의 묶음
        - 1000개 sample, batch size = 20이면 50번의 가중치 갱신
    - epoch : 학습 횟수, 전체 데이터를 얼마나 학습할 것인가?
        - 모든 batch 학습 완료시 1 epoch 진행된것

- 활성화 함수 & 손실 함수
    > [활성화함수](#활성화함수)
    > - [계단함수](#계단함수)
    > - [시그모이드함수](#시그모이드함수)
    > - [렐루함수](#렐루함수)
    > - [소프트맥스함수](#소프트맥스함수)
    >
    > [손실함수](#손실함수)
    > - [오차제곱합](#오차제곱합)
    > - [크로스엔트로피](#크로스엔트로피)


    - 신경망 학습 주요 문제
        - 분류 & 회귀
        - 분류 : 데이터가 어떤 클래스에 속하는가
        - 회귀 : 데이터 수치 예측
    
    - <p id='활성화함수'>활성화 함수</p>

        > - 출력층 : 타겟 = 1, 나머지 = 0
        > - 신경망 통과 값 : 출력층 가중치와 계산 스코어
        > - 출력신호 : 출력층과 비교하는 값(0~1 사이의 값)
        > - score -> 출력신호로 변환하는 함수
        > - 계산한 출력신호를 손실 함수로
        > - 어떤 값을 1로 기준잡을 것인지

        - <p id='계단함수'>계단 함수</p>
            
            ![화면 캡처 2023-05-30 223031](https://github.com/kysssib/Text-Processing/assets/113497500/53964f43-b2cd-440b-bb01-d3a13c880c7c)
            - $h(x) = {0(x\le  0)\brace 1(x>0)}$
            - 잘 안씀
            - 임계값 기준 넘으면 1, 아니면 0

        - <p id='시그모이드함수'>시그모이드 함수</p>
            
            ![화면 캡처 2023-05-30 223031](https://github.com/kysssib/Text-Processing/assets/113497500/6d7d31b5-b5f2-477c-b9a9-dc988d6fb0be)
            - $y=\frac{1}{1+e^{-x}}$
            - 모든 값에 대하여 0~1 사이 값
            - $\displaystyle \lim_{x \to 0}{f(x)}$ 일때 급변하며 
            - $\displaystyle \lim_{x \to _{-}^{+} \infty}{f(x)}$ 일때 정체된다

        - <p id='렐루함수'>렐루 함수</p>

            ![image](https://github.com/kysssib/Text-Processing/assets/113497500/fc5456af-2b0f-4a42-8302-1d5558055669)
            - $h(x) = {0(x\le  0)\brace x(x>0)}$
            - 입력이 0을 넘으면 그 값을 그대로 출력하고, 0이하의 값은 0으로 출력함

        - <p id='소프트맥스함수'>소프트맥스 함수</p>

            - $p_i=\frac{e^{z_i}}{\Sigma^k_{j=1} e^{z_j}} for$ $i = 1,2,...,k$
            - 분자 : 입력 신호의 지수함수
            -  분모 : 모든 입력신호의 지수함수의 합
            - 다만, 여기서 시그마가 전체를 계산하는 값이기 때문에 시간이 오래 걸림 -> softmax에 들어가는 함수 줄임
            - 네거티브 샘플링
                - 타겟 단어 클래스 : 출력 값 1인 단어
                - 타겟 아닌 단어 클래스의 일부 : 출력 값이 0인 단어(네거티브 샘플링)
                - 타겟 1, 네거티브 샘플 2개인 경우
                - $softmax(z) = [\frac{e^{z_1}}{\Sigma^3_{j=1} e^{z_j}}\frac{e^{z_2}}{\Sigma^3_{j=1} e^{z_j}}\frac{e^{z_3}}{\Sigma^3_{j=1} e^{z_j}}]=[p_1,p_2,p_3]$
    
    - <p id='손실함수'>손실 함수</p>

        > - 신경망은 현재 상태를 '하나의 지표'로 표현
        > - 지표를 가장 좋게 만들어 주는 최적 가중치 탐색
        > - '하나의 지표'로 계산하는 것이 손실 함수의 역할
        > - 오차제곱합과 크로스 엔트로피를 주로 사용
        > - 즉, 최적의 가중치 탐색을 위한 오차율 계산 함수
        - <p id='오차제곱합'>오차 제곱합</p>

            - $E=\frac{1}{2}\displaystyle\sum_k(y_k-t_k)^2$
            - $y$ = 신경망 추정 출력값 : 소프트맥스 통과 값(정답일 확률)
            - $t$ = 출력 레이블 값 : 정답 값
            - $k$ = 데이터 차원
            - 이 값이 작을수록 실제값과 추정값이 비슷함
            - $\frac{1}{2}$는 경사하강법 오류의 최소화 위함

        - <p id='크로스엔트로피'>크로스 엔트로피</p>

            - $E=-\displaystyle\sum_k t_k log y_k$
            - $y$ = 신경망 추정 출력값 : 소프트맥스 통과 값(정답일 확률)
            - $t$ = 출력 레이블 값 : 정답 값
            - $k$ = 데이터 차원
            - 정답 원소만 1이므로 정답일 때 추정값의 자연로그값을 계산하는 것
            - 추정값이 정답이면 자연로그는 0, 정답과 가까울수록 0에 가까움
            ![image](https://github.com/kysssib/Text-Processing/assets/113497500/7decbca8-33c6-4d66-9ce2-e6c1fa48d388)
                - x=0 : 로그 값 없음
                - x가 0에 근접 : 로그값은 $-\infty$
                - x가 1에 근접 : 로그값은 0에 근접
                - x가 1 : 로그값은 0

    - 네트워크 가중치 갱신(옵티마이저)
        - 활성화함수 통과 후, 교차 엔트로피 오차 계산 후 가중치 갱신필요
        - 오차가 나타난 방향으로 가중치 갱신

        - 경사하강법
            - 가장 기본적 가중치 갱신 방법
            - 모든 데이터의 오차에 대한 미분을 기울기 계산
            - 이 오차를 줄이는 방향으로 가중치 갱신
            - 모든 데이터 대해 계산 -> 시간 소모
            - 빠른 처리를 위해 무작위 샘플링 통해 경사하강법 진행(SGD)
        - 다양한 옵티마이저
             - 모멘텀
             - 아다그래드
             - 알엠에스프롭
             - 아담
</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 13. TMDB 실습
<details>

1. 데이터 전처리
    - 드라이브 마운트, 불러오기 및 null행 삭제
    ```python
    #구글드라이브 마운트
    from google.colab import drive
    import os
    drive.mount('/content/drive') #구글드라이브 마운트 위치

    import pandas as pd
    #구글 드라이브 내 다운로드 받은 csv 파일의 디렉토리 위치 설정
    path_dir = './drive/MyDrive/Colab Notebooks/sampledata/' 
    file = path_dir+'tmdb_5000_movies.csv' #다운로드 받은 파일

    df = pd.read_csv(file)
    movies = df[['original_title', 'overview']]
    movies = movies.dropna(axis=0)              #NaN 값이 있는 행 삭제(axis=1 >> 열 삭제)
    print(movies.shape)
    ```
2. 문장 토큰화
    - overview의 모든 문장을 nomalized_text에 저장 후 result에 각 문장 토큰화 해 저장
    ```python
    import re
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    nltk.download('punkt')

    overview_text = movies['overview']

    normalized_text = []
    for string in overview_text:
        tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
        #문자 소문자화 후 소문자,숫자 외의 모든 문자를 ' '으로 변경
        normalized_text.append(tokens)

    # 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
    result = [word_tokenize(sentence) for sentence in normalized_text]
    print('총 샘플의 개수 : {}'.format(len(result)))

    ###출력결과
    총 샘플의 개수 : 4800
    ```
3. Word2Vec 파라미터 셋팅 및 모델 생성
    
    모델 파라미터 값
    1. vector_size : 히든레이어 크기 -> 밀집 벡터의 크기 의미(100~300사이)
    2. window : 맥락의 크기, 학습단어 좌/우 단어 선택 수
    3. min_count : 최소 빈도수 이하 단어 모델 학습에 배제
    4. workers : 모델 학습 시 쓰레드 개수
    5. sg : 0이면 CBOW, 1이면 SkipGram
    6. hs : 0이면 네거티브 샘플링, 1이면 softmax(Default 0)
    7. negative : 0보다 크면 네거티브샘플링의 'noise words' 갯수로 활용(Default 5)
    8. alpha : learning rate(Default 0.025)
    9. epochs : 전체 코퍼스 반복횟수(Default 5)
    10. batch_words : 단어의 배치 크기(Default 10000)
    
- TDMB 데이터 셋 Word2Vec모델 생성, 저장
    ```python 
    from gensim.models import Word2Vec
    from gensim.models import KeyedVectors

    model = Word2Vec(sentences=result, vector_size=100, window=2, min_count=5, workers=4, sg=0)
    #계산 된 모델을 파일경로에 tmdb.model로 저장
    model.save(path_dir+'tmdb.model')
    ```
- TMDB 모델 로드, 확인 
    ```python 
    from gensim.models import Word2Vec
    
    #저장된 모델을 로딩
    model = Word2Vec.load(path_dir+'tmdb.model')

    from gensim.models import KeyedVectors
    import numpy as np

    np.set_printoptions(precision=4, suppress=True)
    #단어 action의 임베딩 데이터 확인
    vector = model.wv['action']
    print(len(vector)) #히든 레이어가 100이므로 100크기로 임베딩됨
    print(vector)
    #------------------------------
    #출력결과
    100
    [ 0.0022  0.4948 -0.1071 -0.0132 -0.4161  0.2684 -0.1393 -0.3209 -0.1255
    -0.0001  0.1339 -0.3903  0.1762 -0.0301  0.1442  0.2307  0.3176  0.306
    0.2345 -0.5239 -0.5775  0.5355 -0.0685  0.0601 -0.1139 -0.1307 -0.0753
    0.21   -0.0488  0.0438 -0.4096 -0.3975 -0.0171 -0.0903  0.0987 -0.0391
    0.0172 -0.203   0.256   0.2358  0.1464  0.2245 -0.1092 -0.1938 -0.2323
    0.1351  0.3515  0.201   0.1164  0.2019  0.2507  0.4502 -0.0073 -0.2095
    -0.2257 -0.0936  0.3458 -0.2447  0.1566 -0.3166 -0.1323 -0.3226 -0.5186
    0.0992 -0.1003 -0.0626  0.2178  0.0279 -0.0367  0.1141  0.5313 -0.2391
    -0.2287 -0.1824 -0.223   0.0951 -0.1431 -0.2401  0.0844 -0.1622  0.3878
    -0.1524 -0.1482  0.0566  0.0891  0.3509  0.2894  0.0924  0.1543  0.261
    -0.3001 -0.0677 -0.1513 -0.2091 -0.7584 -0.7606 -0.1204  0.0177 -0.3488
    0.1927]
    ```
- wv.most_similar 함수
    ```python
    sims = model.wv.most_similar('action', topn=10)
    #action과 코사인 유사도가 높은 10개의 단어 출력
    print(sims)

    ###출력결과
    [('adventure', 0.9994416236877441), ('state', 0.9993367791175842), 
    ('east', 0.9992945790290833), ('top', 0.9992831349372864),
    ('coming', 0.9992748498916626), ('desert', 0.9992640614509583),
    ('pair', 0.9992302060127258), ('community', 0.9992255568504333),
    ('police', 0.9991926550865173), ('early', 0.9991670846939087)]
    ```

- KeyedVectors Save & Load

    해당 결과를 KeyedVectors에 분리해 저장
    ```python
    w2v_tmdb = path_dir+'w2v_tmdb'

    word_vectors = model.wv
    word_vectors.save(w2v_tmdb)

    wv = KeyedVectors.load(w2v_tmdb, mmap='r')
    print(wv.vectors.shape)

    ###출력결과
    (5549, 100)
    ```

- KeyedVectors 활용

    <details>
    <summary>most_similar</summary>

    ```python
    result = wv.most_similar(positive=['action', 'man'], negative=['woman'])
    most_similar_key, similarity = result[0]
    print(f"{most_similar_key}: {similarity:.4f}")

    ###출력결과
    secret: 0.9910
    ```
    </details>

    <details>
    <summary>similar_by_word</summary>

    ```python
    result = wv.similar_by_word("man")
    most_similar_key, similarity = result[0]
    print(f"{most_similar_key}: {similarity:.4f}")

    ###출력결과
    woman: 0.9862
    ```
    man과 유사한 단어로 woman이 나옴
    </details>

    <details>
    <summary>wmdistance,distance</summary>

    ```python
    batman = 'action movie is very fun and easy'.lower().split()
    action = 'action movie is very fun and easy'.lower().split()
    distance = wv.wmdistance(batman, action)
    print(f"{distance:.4f}")

    ###출력결과
    0.0000
    ```
    두 문장의 거리를 계산한다<br>
    거리가 낮을수록 유사하다
    
    ```python
    distance = wv.distance("batman", "batman")
    print(f"{distance:.1f}")

    ###출력결과
    0.0
    ```
    위와 다른 점은 단어 사이의 거리를 계산한다.
    </details>

    <details>
    <summary>n_similarity</summary>

    ```python
    similarity = wv.n_similarity(['action', 'batman'], ['romance', 'comic'])
    print(f"{similarity:.4f}")

    ###출력결과
    0.9992
    ```
    여러 단어의 유사도를 계산한다
    </details>



</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 14. 네이버 영화 실습
<details>

- 영화 자료는 [여기서](#https://github.com/e9t/nsmc)(https://github.com/e9t/nsmc) 가지고 온다

1. 데이터 로딩과 전처리
    ```python
    #구글드라이브 마운트
    from google.colab import drive
    import os
    drive.mount('/content/drive') #구글드라이브 마운트 위치

    import pandas as pd
    #구글 드라이브 내 다운로드 받은 무비데이터 파일의 디렉토리 위치 설정
    path_dir = './drive/MyDrive/Colab Notebooks/sampledata/' 
    file = path_dir+'ratings.txt'  #https://github.com/e9t/nsmc에서 다운로드

    data = pd.read_table(file)

    print(data.shape)
    print(data.head(5))
    data = data.dropna(how = 'any') # 빈 행(NULL값) 제거
    print(data.shape)

    ###출력결과
    (200000, 3)
            id                                           document  label
    0   8112052                                어릴때보고 지금다시봐도 재밌어요ㅋㅋ      1
    1   8132799  디자인을 배우는 학생으로, 외국디자이너와 그들이 일군 전통을 통해 발전해가는 문화산...      1
    2   4655635               폴리스스토리 시리즈는 1부터 뉴까지 버릴께 하나도 없음.. 최고.      1
    3   9251303  와.. 연기가 진짜 개쩔구나.. 지루할거라고 생각했는데 몰입해서 봤다.. 그래 이런...      1
    4  10067386                        안개 자욱한 밤하늘에 떠 있는 초승달 같은 영화.      1
    (199992, 3)
    ```
2. 형태소 분석
    ```python
    !curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash
    ```
    해당 코드 입력 후 불용어 파일을 로딩해 불용어 리스트에 저장

    ```python
    #불용어 로딩
    file = path_dir+'kstopwords100.txt' #불용어 처리 파일

    stopwords = pd.read_table(file, header=None)
    print(stopwords[:5])
    stopwords = stopwords[:][0].to_list()

    len(stopwords)

    ###출력결과
    0    1         2
    0  이  VCP  0.018280
    1  있   VA  0.011699
    2  하   VV  0.009774
    3  것  NNB  0.009733
    4  들  XSN  0.006898
    100
    ```
    후에 형태소 분석기 호출
    ```python
    from konlpy.tag import Okt

    okt = Okt()
    tokenized_data = []
    for sentence in data['document']:
        tok_sent = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sent = [word for word in tok_sent if not word in stopwords] # 불용어 제거
        tokenized_data.append(stopwords_removed_sent)
    
    #형태소분석기과정 거친 자료 피클로 저장
    import pickle

    with open(path_dir+"naver_mv_tokenized_data.pickle","wb") as f:
        pickle.dump(tokenized_data, f)

    
    ```

3. 분석 파일 로드 후 Word2Vec 실습
    ```python
    #피클 파일 불러오기
    import pickle

    path_dir = './drive/MyDrive/Colab Notebooks/sampledata/'

    with open(path_dir+"naver_mv_tokenized_data.pickle","rb") as f:
        tokenized_data = pickle.load(f)

    #리뷰데이터 분포
    print('리뷰의 최대 길이 :',max(len(d) for d in tokenized_data))
    print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))

    import matplotlib.pyplot as plt

    plt.hist([len(d) for d in tokenized_data], bins=50)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    plt.show()

    ##출력결과
    [['어리다', '보고', '보다', '재밌다', 'ㅋㅋ'], 
    ['디자인', '을', '배우다', '학생', '으로', ',', '외국', '디자이너', '와', '일군', '전통', '을', '통해', '발전', '하다', '문화', '산업', '부럽다', '.', '우리나라', '에서도', '어렵다', '시절', '에', '끝', '까지', '열정', '을', '지키다', '노라노', '같다', '전통', '있다', '저', '와', '같다', '꿈', '을', '꾸다', '이루다', '나가다', '있다', '에', '감사하다', '.'],
    ['폴리스스토리', '시리즈', '는', '1', '부터', '뉴', '까지', '버리다', '도', '없다', '..', '최고', '.']]
    ```
    ![image](https://github.com/kysssib/Text-Processing/assets/113497500/73dd2762-5b30-4a3c-b4c2-9a33cd796b1e)
    
    가로 : 샘플의 길이<br>
    세로 : 해당 샘플의 숫자

    모델 구축
    ```python
    from gensim.models import Word2Vec
    from gensim.models import KeyedVectors

    model = Word2Vec(sentences=tokenized_data,
                    vector_size=100, window=5, min_count=5,
                    workers=4, sg=1)
    
    #단어수 확인
    model.wv.vectors.shape

    ###출력결과
    (17564, 100)
    ```
4. 모델 활용
    <details><summary>most_similar</summary>
    
    ```python
    model_result = model.wv.most_similar("정우성")
    print(model_result)

    ###출력결과
    [('황정민', 0.8435378074645996), ('송혜교', 0.8388928174972534), 
    ('장혁', 0.8372812271118164), ('정재영', 0.8354607820510864), 
    ('원빈', 0.8349665403366089), ('남상미', 0.8325139880180359), 
    ('이민정', 0.8314265608787537), ('심은경', 0.8306748867034912), 
    ('김창완', 0.8292453289031982), ('이범수', 0.8287127017974854)]

    ##아래 출력결과는 CBOW로 진행한 모델로 정우성을 검색해본 내용입니다.
    [('김혜수', 0.8736814856529236), ('주상욱', 0.8694123029708862),
    ('이종석', 0.858771800994873), ('장혁', 0.8580711483955383), 
    ('송승헌', 0.8483266830444336), ('정재영', 0.8478938341140747), 
    ('하지원', 0.8471293449401855), ('임수정', 0.8444806337356567), 
    ('손예진', 0.8371105790138245), ('공효진', 0.8356135487556458)]

    model_result = model.wv.most_similar("액션")
    print(model_result)

    ###출력결과
    [('액션씬', 0.7664161920547485), ('격투', 0.7317063212394714), 
    ('격투씬', 0.7102022171020508), ('총격', 0.7101128697395325), 
    ('무협', 0.7076876163482666), ('레이싱', 0.7047789096832275), 
    ('무술', 0.6988150477409363), ('첩보', 0.6942819356918335), 
    ('추격', 0.6899939775466919), ('코메', 0.6888874769210815)]

    ##아래 출력결과는 CBOW로 진행한 모델로 액션을 검색해본 내용입니다
    [('스케일', 0.7097840309143066), ('코믹', 0.6931434869766235), 
    ('액션씬', 0.6928027868270874), ('볼거리', 0.6472076177597046), 
    ('긴장감', 0.6425592303276062), ('화끈하다', 0.6276863217353821), 
    ('그래픽', 0.6182790994644165), ('무술', 0.6093198657035828), 
    ('화려하다', 0.6072039008140564), ('스릴러', 0.6064084768295288)]
    ```
    Skip-Gram으로 학습한 모델의 테스트
    </details>
    <details><summary>KeyedVectors : save_word2vec_format & load_word2vec_format</summary>
    
    ```python
    w2v_vectors = path_dir+'naver_w2v'
    model.wv.save_word2vec_format(w2v_vectors) # 모델 저장
    wv = KeyedVectors.load_word2vec_format(w2v_vectors) # 모델 로드
    print(type(wv))

    result = wv.most_similar("한석규")
    print(result)

    ###출력결과
    gensim.models.keyedvectors.Word2VecKeyedVectors

    [('엄정화', 0.850074291229248), ('김명민', 0.8498216867446899), 
    ('디카프리오', 0.8420839905738831), ('문소리', 0.8418601155281067), 
    ('안성기', 0.841203510761261), ('신들리다', 0.8388416767120361), 
    ('전도연', 0.8387709259986877), ('설경구', 0.8325169682502747), 
    ('공리', 0.8314179182052612), ('차승원', 0.8303262591362)]
    ```
    여기에서 활용된 벡터 저장 방식은 save()와 load()를 사용하지 않음
    
    save, load 함수는 KeyedVectors 객체 자체를 저장하고 로딩하여 추가학습 등을 진행할 수 있는 반면, 여기서 사용한 save_word2vec_format과 load_word2vec_format은 ‘original C word2vec-tool format’으로 저장되어 추가학습 등을 진행할 수 없는 포멧

    출력결과를 살펴보면 wv가 Word2VecKeyedVectors인 것을 확인할 수 있다.
    </details>

</details>

<div style="text-align: right">

[목차](#목차)
</div>

---

## 15. 사전 임베딩 자료 실습
<details>

1. 영어 사전학습 word2vec 임베딩 로딩

    자료는 [여기서](#https://code.google.com/archive/p/word2vec/)(https://code.google.com/archive/p/word2vec/) 가져온다

    ```python
    from gensim.models import KeyedVectors

    w2v_eng_model = KeyedVectors.load_word2vec_format
        ('GoogleNews-vectors-negative300.bin.gz', binary=True)

    print(w2v_eng_model.vectors.shape)

    ##출력
    (3000000, 300)
    ```

2. 한글 사전학습 word2vec 임베딩

    자료는 [여기서](#https://github.com/Kyubyong/wordvectors)(https://github.com/Kyubyong/wordvectors) 가져온다<br>
    약 30개국 언어의 임베딩 자료를 구할 수 있다

    ```python
    from gensim.models.word2vec import Word2Vec
    import numpy

    w2v_kor_model = Word2Vec.load('ko.bin')
    ```

3. Naver 영화평 벡터 활용하기

    - 구글 드라이브 마운트와 네이버 영화평 임베딩 벡터 로딩
        ```python
        from google.colab import drive
        import os
        drive.mount('/content/drive') #구글드라이브 마운트 위치
        #구글 드라이브 내 모델 데이터 위치 설정
        path_dir = './drive/MyDrive/Colab Notebooks/sampledata/'

        from gensim.models import Word2Vec
        from gensim.models import KeyedVectors

        w2v_vectors = path_dir+'naver_w2v'
        wv = KeyedVectors.load_word2vec_format(w2v_vectors) # 모델 로드

        print(wv.vectors.shape)

        ### 출력결과
        (17564, 100)
        ```

    - 임베딩 벡터 활용

        형태소분석기 및 불용어 로딩
        ```python
        !curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash
        ```
        ```python
        import pandas as pd

        #불용어 로딩
        file = path_dir+'kstopwords100.txt' 

        stopwords = pd.read_table(file, header=None)
        print(stopwords[:5])
        stopwords = stopwords[:][0].to_list()

        len(stopwords)

        ###출력결과
        0    1         2
        0  이  VCP  0.018280
        1  있   VA  0.011699
        2  하   VV  0.009774
        3  것  NNB  0.009733
        4  들  XSN  0.006898
        100
        ```
        샘플 문서 형태소 분석 해당 sentence변수에 든 문장은 임의 문장이다
        ```python
        sentence = ['베테랑이 제일 재미있었다.', '황정민 연기는 정말 끝내준다', '새로나온 재미있는 영화 없나?',
                    '우리나라에서 이런 영화를 만들줄 몰랐다.', '연기 못하는 배우도 잘나가는구나']
                    
        from konlpy.tag import Okt

        okt = Okt()
        tokenized_data = []
        for sent in sentence:
            tok_sent = okt.morphs(sent, stem=True) # 토큰화
            stopwords_removed_sent = [word for word in tok_sent if not word in stopwords]
            tokenized_data.append(stopwords_removed_sent)

        print(tokenized_data)

        ### 출력결과
        [['베테', '랑', '제일', '재미있다', '.'], 
        ['황정민', '연기', '는', '정말', '끝내다'], 
        ['새롭다', '나오다', '재미있다', '영화', '없다', '?'], 
        ['우리나라', '에서', '이렇다', '영화', '를', '만들다', '모르다', '.'], 
        ['연기', '못', '하다', '배우다', '잘나가다']]
        ```
        케라스를 활용한 토크나이저 및 사전 구축
        ```python
        import numpy as np
        from keras.preprocessing.text import Tokenizer

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tokenized_data)

        vocab_size = len(tokenizer.word_index) + 1
        print('단어 개수 :',vocab_size)
        print(tokenizer.word_index)

        ###출력결과
        단어 개수 : 26
        {'재미있다': 1, '.': 2, '연기': 3, '영화': 4, '베테': 5, '랑': 6, '제일': 7, '황정민': 8, 
        '는': 9, '정말': 10, '끝내다': 11, '새롭다': 12, '나오다': 13, '없다': 14, '?': 15, 
        '우리나라': 16, '에서': 17, '이렇다': 18, '를': 19, '만들다': 20, '모르다': 21, '못': 22,
        '하다': 23, '배우다': 24, '잘나가다': 25}
        ```
        단어 임베딩 만들기
        ```python
        embedding_vec = np.zeros((vocab_size, 100)) #네이버 단어임베딩 벡터 길이 100
        print(np.shape(embedding_vec))

        ###출력결과
        (26, 100)
        ```
        wv에서 임베딩 자료를 가져올 함수 생성(함수는 해당 단어의 임베딩 벡터를 반환, 없으면 None)
        ```python
        def get_vector(word):
            if word in wv:
                return wv[word]
            else:
                return None
        #사용법
        for word, index in tokenizer.word_index.items():
            vector = get_vector(word)
            if vector is not None:
                embedding_vec[index] = vector
        print(embedding_vec[1])

        ###출력결과
        [-0.90842527  0.24096963  0.07952304 -0.07481565 -0.23117378 -0.45360285
        -0.02481424  0.04972698  0.29619971  0.33146238 -0.38025373 -0.12967564
        -0.17546327 -0.1940318   0.11065004 -0.46046361 -0.09199833 -0.37428865
        -0.70732927  0.23981455  0.2995061   0.06040657 -0.02012163 -0.06818607
        0.00870245  0.02101303  0.01328442 -0.03182668  0.01928779  0.13133152
        -0.26575708  0.03603864 -0.23061009  0.29820696  0.31748411  0.21167077
        -0.02976821 -0.11636379  0.17527997 -0.29821178 -0.03002749 -0.38743222
        -0.44537312 -0.37033185 -0.06709219  0.1261839   0.15775077  0.4391903
        -0.00690586 -0.24710579 -0.54626775  0.01758383 -0.07784662  0.36384881
        0.26519459 -0.34380242  0.04659332  0.22215968  0.04436301 -0.65513325
        -0.10668164 -0.52401096 -0.32037759 -0.06596681 -0.10190149 -0.07385413
        -0.10744798  0.4143092  -0.00781951  0.39623666 -0.29232049  0.25385565
        0.50013185  0.47848335 -0.00569391 -0.04954932 -0.04758142  0.14865264
        0.1752449   0.25221092 -0.0458079   0.27935067 -0.05026009  0.17818071
        -0.29584485 -0.16940355  0.18712966  0.245657   -0.29228112  0.38417351
        0.24896795  0.37891448  0.44761166  0.13117118  0.17999749 -0.02614649
        -0.58532465 -0.15918075  0.1183216   0.28335199]
        ```
        해당 벡터가 네이버 모델과 비슷한지 비교함
        ```python
        print(wv['재미있다'])

        ###출력결과
        [-0.9084253   0.24096963  0.07952304 -0.07481565 -0.23117378 -0.45360285
        -0.02481424  0.04972698  0.2961997   0.33146238 -0.38025373 -0.12967564
        -0.17546327 -0.1940318   0.11065004 -0.4604636  -0.09199833 -0.37428865
        -0.7073293   0.23981455  0.2995061   0.06040657 -0.02012163 -0.06818607
        0.00870245  0.02101303  0.01328442 -0.03182668  0.01928779  0.13133152
        -0.26575708  0.03603864 -0.23061009  0.29820696  0.3174841   0.21167077
        -0.02976821 -0.11636379  0.17527997 -0.29821178 -0.03002749 -0.38743222
        -0.44537312 -0.37033185 -0.06709219  0.1261839   0.15775077  0.4391903
        -0.00690586 -0.24710579 -0.54626775  0.01758383 -0.07784662  0.3638488
        0.2651946  -0.34380242  0.04659332  0.22215968  0.04436301 -0.65513325
        -0.10668164 -0.52401096 -0.3203776  -0.06596681 -0.10190149 -0.07385413
        -0.10744798  0.4143092  -0.00781951  0.39623666 -0.2923205   0.25385565
        0.50013185  0.47848335 -0.00569391 -0.04954932 -0.04758142  0.14865264
        0.1752449   0.25221092 -0.0458079   0.27935067 -0.05026009  0.17818071
        -0.29584485 -0.16940355  0.18712966  0.245657   -0.29228112  0.3841735
        0.24896795  0.37891448  0.44761166  0.13117118  0.17999749 -0.02614649
        -0.58532465 -0.15918075  0.1183216   0.283352  ]
        ```
</details>

<div style="text-align: right">

[목차](#목차)
</div>

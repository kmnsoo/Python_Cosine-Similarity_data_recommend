# pandas라이브러리 불러오기
import pandas as pd

# 준비한 csv파일 불러오기
train = pd.read_csv('movies_metadata/movies_metadata.csv') 

# csv파일 0~9번까지 출력
#print(train.head(10)) 

# 피쳐가 title과 vote_average만 해당하게 객체 생성
X_train = train[['title','vote_average']]
#Y_train = train['vote_average']
#print(X_train.head(3), Y_train.head(3))

# vote_average피쳐 기준으로 오름차순 (sort정렬)
#X_train.sort_values(by=['vote_average'], axis=0)

# vote_average피쳐 기준으로 내림차순 (ascending=False)
X_train.sort_values(by=['vote_average'], axis=0, ascending=False)

# 정렬한 피쳐들 10개만 출력
#print(X_train.sort_values(by=['vote_average'], axis=0).head(10))

# 정렬한 피쳐들 10개만 출력
print(X_train.sort_values(by=['vote_average'], axis=0, ascending=False).head(10))

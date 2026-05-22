# 🚀 경마 다크호스 대시보드 배포 가이드

이 문서는 완성된 Streamlit 대시보드를 GitHub와 Streamlit Community Cloud를 통해 외부로 공유하는 방법을 설명합니다.

## 1. 사전 준비
- [GitHub 계정](https://github.com/)이 필요합니다.
- [Git](https://git-scm.com/)이 설치되어 있어야 합니다.

## 2. `.gitignore` 설정
불필요한 파일이 공유되지 않도록 프로젝트 루트에 `.gitignore` 파일을 확인하거나 생성합니다.
```text
.venv/
__pycache__/
models/*.pkl
images/
reports/
.DS_Store
```

## 3. GitHub 레포지토리 업로드
터미널에서 아래 명령어를 순서대로 실행합니다.

```bash
# 1. 로컬 저장소 초기화
git init

# 2. 파일 스테이징 및 커밋
git add .
git commit -m "feat: 경마 다크호스 예측 대시보드 구현 완료"

# 3. 브랜치 설정 및 원격 저장소 연결
git branch -M main
# [주의] 아래 URL은 본인의 GitHub 레포지토리 주소로 변경하세요.
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 4. 푸시
git push -u origin main
```

## 4. Streamlit Cloud 배포
1. [Streamlit Community Cloud](https://share.streamlit.io/) 접속 및 로그인.
2. **[Create app]** 클릭.
3. 해당 GitHub 레포지토리 선택.
4. **Main file path**를 `app.py`로 지정.
5. **[Deploy!]** 클릭 후 약 1~3분 대기.

## 5. 공유 및 업데이트
- 배포가 완료되면 생성된 URL(예: `https://darkhorse-ai.streamlit.app`)을 공유하면 됩니다.
- 이후 코드를 수정하고 `git push`를 하면 배포된 사이트도 자동으로 업데이트됩니다.

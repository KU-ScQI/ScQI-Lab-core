# ScQI-Lab-core
Core repository for experiments

## Git setup
이 저장소를 클론한 뒤, 터미널에서 아래 커맨드를 한 번 실행하세요.

```bash
pip install pre-commit
pre-commit install
```

## 규칙
- 측정 데이터 (.h5, .npy 등) 는 업로드하지 않습니다
- main에 직접 push하지 않고, 브랜치 나눠서 PR
- PR 실패시 머지하지 않기

## 참고
- Jupyter Notebook을 업로드할 경우, 아웃풋이 남아있으면 용량을 잡아먹기 때문에 pre-commit 과정에서 아웃풋을 지워줍니다.
- 그 경우 commit fail이 떠도 놀라지 마세요.

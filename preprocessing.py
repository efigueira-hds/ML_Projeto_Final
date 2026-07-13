"""
Pré-processamento partilhado pelos notebooks de modelação.

Este módulo centraliza a lógica que estava copiada em vários notebooks
(feedback #6 — conteúdo duplicado) e separa dois tipos de transformação:

* Determinísticas / de domínio — seguras de aplicar ANTES do split porque
  não aprendem parâmetros a partir dos dados (999 -> NaN, recodificação de
  Sexo, fórmula do IMC, reconstrução do WAtotal_0, remoção de casos sem
  Grupo_pre e remoção das variáveis pós-operatórias). Vivem em `carregar_dados`.

* Aprendidas a partir dos dados — imputação pela mediana, scaling e SMOTE.
  Estas TÊM de ser ajustadas apenas com o treino, por isso vivem dentro de
  uma `Pipeline`/`ColumnTransformer` (feedback #1 e #5). Assim, em cada fold
  da validação cruzada só o treino é usado para as estimar.

Nota: os notebooks devem carregar os dados por aqui (a partir do Excel
original), e NÃO pelo `ortho_eda_clean.csv`, porque esse ficheiro já tem a
imputação feita sobre todas as linhas (o que provocaria data leakage). O CSV
continua válido apenas para exploração.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CAMINHO_DADOS = "ML_Ortopedia_CPAK.xlsx"
TEST_SIZE = 0.30

# Variáveis com fuga de informação pós-operatória: nunca entram nos modelos.
COLUNAS_LEAKAGE = [
    "Fle_90", "EVA_90", "PM6_90", "WD_90", "WR_90",
    "WAtotal_90", "WT_90", "Satisfação", "Grupo_pos",
]

# Grupo_pre é a única categórica com mais de 2 categorias -> one-hot encoding.
# Sexo já é binária (0/1) e é tratada como numérica.
COLUNAS_CATEGORICAS = ["Grupo_pre"]


def carregar_dados(caminho=CAMINHO_DADOS):
    """Carrega o Excel e aplica só transformações determinísticas/de domínio.

    Devolve (X, y). NÃO faz imputação nem scaling — esses passos aprendem dos
    dados e ficam dentro da pipeline, ajustada apenas com o treino. Os NaN
    remanescentes são mantidos de propósito em X, para serem tratados lá.
    """
    df = pd.read_excel(caminho)
    df.columns = df.columns.str.strip()  # corrige o nome "Idade " (com espaço)

    # 999 é um sentinela de valor em falta em toda a base (nunca é um valor
    # clínico legítimo em nenhuma das variáveis) -> substituir por NaN.
    df = df.replace(999, np.nan)

    # Casos sem Grupo_pre não permitem definir o target -> remover.
    df = df.dropna(subset=["Grupo_pre"])
    df["Grupo_pre"] = df["Grupo_pre"].astype("int64")

    # Recodificação de Sexo (1/2 -> 0/1) para não induzir hierarquia no modelo.
    df["Sexo"] = df["Sexo"].replace({1: 0, 2: 1})

    # Target: o grupo CPAK mudou entre o pré e o pós-operatório?
    y = (df["Grupo_pre"] != df["Grupo_pos"]).astype(int)

    # WR_0 tem escala 0-8; valores acima são erros de preenchimento -> NaN.
    df.loc[df["WR_0"] > 8, "WR_0"] = np.nan

    # IMC recalculado a partir de Peso e Altura (fórmula, linha a linha).
    m_imc = df["Peso"].notna() & df["Altura_cm"].notna() & (df["Altura_cm"] > 0)
    df.loc[m_imc, "IMC"] = df.loc[m_imc, "Peso"] / ((df.loc[m_imc, "Altura_cm"] / 100) ** 2)

    # WAtotal_0 reconstruído a partir de WT_0, WR_0 e WD_0 (linha a linha).
    m_wa = df["WAtotal_0"].isna()
    df.loc[m_wa, "WAtotal_0"] = df.loc[m_wa, "WT_0"] - (df.loc[m_wa, "WR_0"] + df.loc[m_wa, "WD_0"])

    # Remover as variáveis de fuga (pós-operatório) e o Grupo_pos.
    df = df.drop(columns=COLUNAS_LEAKAGE)

    X = df.copy()
    return X, y


def dividir_treino_teste(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split estratificado único (feedback #1). O teste é tocado uma só vez."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def construir_preprocessador(X):
    """ColumnTransformer com a imputação (mediana) e o one-hot encoding.

    Estes passos aprendem dos dados e, dentro de uma pipeline, são ajustados
    apenas com o treino de cada fold (evita data leakage).
    """
    cols_cat = [c for c in COLUNAS_CATEGORICAS if c in X.columns]
    cols_num = [c for c in X.columns if c not in cols_cat]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), cols_num),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False), cols_cat),
        ]
    )


def construir_pipeline(modelo, X, escalar=False, usar_smote=False,
                       random_state=RANDOM_STATE):
    """Constrói a pipeline de pré-processamento + modelo.

    * escalar=True acrescenta StandardScaler (útil p.ex. na Regressão Logística).
    * usar_smote=True usa a Pipeline do imbalanced-learn, para o SMOTE ser
      aplicado só no fit (folds de treino) e nunca à validação/teste
      (feedback #4). O SMOTE entra depois do pré-processamento, já com dados
      numéricos e imputados.
    """
    passos = [("prep", construir_preprocessador(X))]
    if escalar:
        passos.append(("scaler", StandardScaler()))
    if usar_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        passos.append(("smote", SMOTE(random_state=random_state)))
        passos.append(("modelo", modelo))
        return ImbPipeline(passos)
    passos.append(("modelo", modelo))
    return Pipeline(passos)


def _scores(pipeline, X):
    """Probabilidades da classe positiva (ou decision_function em alternativa)."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def avaliar_teste(pipeline, X_test, y_test, titulo="", plot=True):
    """Avaliação final no teste: matriz de confusão + report + AUC.

    A AUC é calculada a partir de PROBABILIDADES (feedback #2), não das
    classes previstas.
    """
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score)

    y_pred = pipeline.predict(X_test)
    y_score = _scores(pipeline, X_test)

    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_score)
    print("AUC (probabilidades):", round(auc, 4))

    cm = confusion_matrix(y_test, y_pred)
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest")
        plt.xlabel("Classe Prevista")
        plt.ylabel("Classe Real")
        plt.title("Matriz de confusão" + (f" - {titulo}" if titulo else ""))
        plt.show()
    return {"auc": auc, "matriz_confusao": cm, "y_pred": y_pred, "y_score": y_score}


def validacao_cruzada_treino(pipeline, X_train, y_train, n_splits=5,
                             random_state=RANDOM_STATE):
    """Validação cruzada estratificada usando APENAS o treino (feedback #1).

    Devolve as previsões e as probabilidades out-of-fold e imprime o report e
    a AUC (calculada a partir das probabilidades — feedback #2).
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import classification_report, roc_auc_score

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred = cross_val_predict(pipeline, X_train, y_train, cv=cv)
    proba = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                              method="predict_proba")[:, 1]
    print(classification_report(y_train, pred, digits=4))
    print("AUC CV (probabilidades):", round(roc_auc_score(y_train, proba), 4))
    return pred, proba

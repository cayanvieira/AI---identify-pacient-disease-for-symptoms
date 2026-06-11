import { useState, useEffect, useRef } from 'react';
import './App.css';
import pacients from '../data/pacients.json';
import diseases from '../data/diseases.json';
// Nova estrutura de sintomas: Value fixo para a IA e Labels para a interface
const LISTA_SINTOMAS = [
  { value: "fever", labelPt: "Febre", labelEn: "Fever" },
  { value: "dry_cough", labelPt: "Tosse Seca", labelEn: "Dry Cough" },
  { value: "headache", labelPt: "Dor de Cabeça", labelEn: "Headache" },
  { value: "sore_throat", labelPt: "Dor de Garganta", labelEn: "Sore Throat" },
  { value: "shortness_of_breath", labelPt: "Falta de Ar", labelEn: "Shortness of Breath" },
  { value: "fatigue", labelPt: "Fadiga", labelEn: "Fatigue" },
  { value: "body_aches", labelPt: "Dores no Corpo", labelEn: "Body Aches" },
  { value: "loss_of_taste_smell", labelPt: "Perda de Paladar/Olfato", labelEn: "Loss of Taste/Smell" },
  { value: "runny_nose", labelPt: "Coriza", labelEn: "Runny Nose" },
  { value: "nausea", labelPt: "Nausea", labelEn: "Nausea" },
  { value: "dizziness", labelPt: "Tontura", labelEn: "Dizziness" },
  { value: "abdominal_pain", labelPt: "Dor Abdominal", labelEn: "Abdominal Pain" }
];

const TEXTOS = {
  pt: {
    titulo: "Triagem Inteligente 🩺",
    subtitulo: "Insira os dados do paciente para análise preliminar de estudo.",
    secao1: "1. Dados do Paciente",
    nome: "Nome Completo:",
    idade: "Idade:",
    peso: "Peso (kg):",
    fumantePergunta: "O paciente é fumante?",
    sim: "Sim",
    nao: "Não",
    secao2: "2. Sintomas Apresentados",
    selectPlaceholder: "-- Selecione um sintoma --",
    btnAdicionar: "Adicionar",
    sintomasSelecionados: "Sintomas selecionados",
    listaVazia: "Nenhum sintoma adicionado ainda.",
    btnEnviar: "Analisar Ficha com IA 🚀",
    alerta: "Dados enviados! Abra o Console do Navegador (F12) para ver o JSON com os values padronizados.",
    modalTitulo: "Configuração do Ambiente 🧠",
    modalTexto: "Bem-vindo ao sistema de estudo médico. Antes de preencher a ficha dos pacientes, precisamos inicializar e calibrar os pesos da rede neural da IA.",
    btnTreinar: "Treinar Modelo de IA Agora ⚡",
    loadingTitulo: "Treinando IA...",
    loadingTexto: "Processando banco de dados de sintomas e mapeando padrões biológicos...",
    resultadoConcluido: "Análise Concluída",
    tituloTopMatches: "Top 3 Pacientes com Sintomas Similares",
    textoAnos: "anos",
    textoMatch: "match"
  },
  en: {
    titulo: "Smart Triage 🩺",
    subtitulo: "Enter patient data for preliminary study analysis.",
    secao1: "1. Patient Data",
    nome: "Full Name:",
    idade: "Age:",
    peso: "Weight (kg):",
    fumantePergunta: "Is the patient a smoker?",
    sim: "Yes",
    nao: "No",
    secao2: "2. Symptoms Presented",
    selectPlaceholder: "-- Select a symptom --",
    btnAdicionar: "Add",
    sintomasSelecionados: "Selected symptoms",
    listaVazia: "No symptoms added yet.",
    btnEnviar: "Analyze Case with AI 🚀",
    alerta: "Data sent! Open Browser Console (F12) to view the JSON with standardized values.",
    modalTitulo: "Environment Setup 🧠",
    modalTexto: "Welcome to the medical study system. Before filling out the patient forms, we need to initialize and calibrate the AI's neural network weights.",
    btnTreinar: "Train AI Model Now ⚡",
    loadingTitulo: "Training AI...",
    loadingTexto: "Processing symptom database and mapping biological patterns...",
    resultadoConcluido: "Analysis Completed",
    tituloTopMatches: "Top 3 Patients with Similar Symptoms",
    textoAnos: "years old",
    textoMatch: "match"
  }
};

const symptomsMap = Object.fromEntries(
  LISTA_SINTOMAS.map(symptom => [
    symptom.value,
    {
      pt: symptom.labelPt,
      en: symptom.labelEn
    }
  ])
);

function App() {

  const [showModal, setShowModal] = useState(true);
  const [loadingAI, setLoadingAI] = useState(false);
  const [sintomaSelecionado, setSintomaSelecionado] = useState('');
  const [listaSintomas, setListaSintomas] = useState([]);
  const [hasResult, setHasResult] = useState(false);
  const [idioma, setIdioma] = useState('en');
  const [nome, setNome] = useState('');
  const [idade, setIdade] = useState('');
  const [peso, setPeso] = useState('');
  const [fumante, setFumante] = useState('nao');
  const [pacient, setPacient] = useState({
    name: "",
    age: "",
    weight: "",
    smoker: "",
    symptoms: "",
  });

  const [resultadoAnalise, setResultadoAnalise] = useState([])


  const workerRef = useRef(null);



  useEffect(() => {

    workerRef.current = new Worker(
      new URL('./workers/modelTraining', import.meta.url),
      { type: 'module' }
    );

    workerRef.current.onmessage = (e) => {
      const { action } = e.data;

      if (action === 'TRAIN_COMPLETE') {
        setShowModal(false)
        setLoadingAI(false);
      }
      if (action === 'ANALISED_PACIENT') {
        setResultadoAnalise(e.data.diseaseMatch)
        setHasResult(true);
      }

    }
    return () => {
      workerRef.current.terminate();
    };

  }, []);

  const startTraining = async () => {
    workerRef.current.postMessage({ action: 'START_TRAINING', pacients, diseases });
    setLoadingAI(true)
  };


  const sendPacient = async (e) => {
    e.preventDefault();

    const currentData = {
      name: nome,
      age: idade,
      weight: peso,
      smoker: fumante,
      symptoms: listaSintomas,
    }

    setPacient(currentData)

    workerRef.current.postMessage({ action: 'ANALISING_PACIENT', pacient: currentData });
  }

  const t = TEXTOS[idioma];


  const obterLabelSintoma = (valueDoSintoma) => {
    const sintoma = LISTA_SINTOMAS.find(s => s.value === valueDoSintoma);
    if (!sintoma) return valueDoSintoma;
    return idioma === 'pt' ? sintoma.labelPt : sintoma.labelEn;
  };

  const adicionarSintoma = () => {
    if (sintomaSelecionado && !listaSintomas.includes(sintomaSelecionado)) {
      setListaSintomas([...listaSintomas, sintomaSelecionado]);
      setSintomaSelecionado('');
    }
  };

  const removerSintoma = (valueParaRemover) => {
    setListaSintomas(listaSintomas.filter(value => value !== valueParaRemover));
  };

  return (
    <div className="container">
      {showModal && (
        <div className="modalOverlay">
          <div className="modalCard">
            <div className="containerIdioma">
              <button
                type="button"
                onClick={() => setIdioma('pt')}
                className="btnIdioma"
                style={{ backgroundColor: idioma === 'pt' ? '#2b6cb0' : '#e2e8f0', color: idioma === 'pt' ? '#fff' : '#4a5568' }}
              >
                🇧🇷 PT
              </button>
              <button
                type="button"
                onClick={() => setIdioma('en')}
                className="btnIdioma"
                style={{ backgroundColor: idioma === 'en' ? '#2b6cb0' : '#e2e8f0', color: idioma === 'en' ? '#fff' : '#4a5568' }}
              >
                🇺🇸 EN
              </button>
            </div>
            {!loadingAI ? (
              <>
                <h2 className="modalTitulo">{t.modalTitulo}</h2>
                <p className="modalTexto">{t.modalTexto}</p>
                <button onClick={startTraining} className="btnTreinar">
                  {t.btnTreinar}
                </button>
              </>
            ) : (
              <div className="containerLoading">
                <div className="spinner"></div>
                <h3 className="loadingTitulo">{t.loadingTitulo}</h3>
                <p className="modalTexto">{t.loadingTexto}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* SELEÇÃO DE IDIOMA */}
      <div className="containerIdioma">
        <button
          type="button"
          onClick={() => setIdioma('pt')}
          className="btnIdioma"
          style={{ backgroundColor: idioma === 'pt' ? '#2b6cb0' : '#e2e8f0', color: idioma === 'pt' ? '#fff' : '#4a5568' }}
        >
          🇧🇷 PT
        </button>
        <button
          type="button"
          onClick={() => setIdioma('en')}
          className="btnIdioma"
          style={{ backgroundColor: idioma === 'en' ? '#2b6cb0' : '#e2e8f0', color: idioma === 'en' ? '#fff' : '#4a5568' }}
        >
          🇺🇸 EN
        </button>
      </div>

      <header className="header">
        <h1 className="titulo">{t.titulo}</h1>
        <p className="subtitulo">{t.subtitulo}</p>
      </header>

      <div className="form">

        {/* SEÇÃO 1: Dados Pessoais */}
        <section className="secao">
          <h2 className="tituloSecao">{t.secao1}</h2>

          <div className="grupoInput">
            <label className="label">{t.nome}</label>
            <input
              type="text"
              placeholder={idioma === 'pt' ? "Ex: João Silva" : "e.g. John Doe"}
              value={nome}
              onChange={(e) => setNome(e.target.value)}
              className="input"
              required
            />
          </div>

          <div className="gridDoisCampos">
            <div className="grupoInput">
              <label className="label">{t.idade}</label>
              <input type="number" value={idade} onChange={(e) => setIdade(e.target.value)} className="input" required />
            </div>
            <div className="grupoInput">
              <label className="label">{t.peso}</label>
              <input type="number" value={peso} onChange={(e) => setPeso(e.target.value)} className="input" required />
            </div>
          </div>

          <div className="grupoInput">
            <label className="label">{t.fumantePergunta}</label>
            <div className="containerRadio">
              <label className="labelRadio">
                <input type="radio" name="fumante" value="sim" checked={fumante === 'sim'} onChange={(e) => setFumante(e.target.value)} /> {t.sim}
              </label>
              <label className="labelRadio">
                <input type="radio" name="fumante" value="nao" checked={fumante === 'nao'} onChange={(e) => setFumante(e.target.value)} /> {t.nao}
              </label>
            </div>
          </div>
        </section>

        {/* SEÇÃO 2: Sintomas */}
        <section className="secao">
          <h2 className="tituloSecao">{t.secao2}</h2>

          <div className="linhaSintoma">
            <select
              value={sintomaSelecionado}
              onChange={(e) => setSintomaSelecionado(e.target.value)}
              className="select"
            >
              <option value="">{t.selectPlaceholder}</option>
              {LISTA_SINTOMAS.map((sintoma) => (
                <option key={sintoma.value} value={sintoma.value}>
                  {idioma === 'pt' ? sintoma.labelPt : sintoma.labelEn}
                </option>
              ))}
            </select>

            <button type="button" onClick={adicionarSintoma} className="btnAdicionar">
              {t.btnAdicionar}
            </button>
          </div>

          {/* Lista de Sintomas Adicionados */}
          <div className="containerLista">
            <label className="label">{t.sintomasSelecionados} ({listaSintomas.length}):</label>
            {listaSintomas.length === 0 ? (
              <p className="listaVazia">{t.listaVazia}</p>
            ) : (
              <ul className="lista">
                {listaSintomas.map((value) => (
                  <li key={value} className="itemLista">
                    <span>{obterLabelSintoma(value)}</span>
                    <button type="button" onClick={() => removerSintoma(value)} className="btnRemover">✕</button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        <button onClick={sendPacient} type="submit" className="btnEnviar">
          {t.btnEnviar}
        </button>

      </div>

      {hasResult && (
        <div className="diagnosis-container">
          {resultadoAnalise.map((disease, index) => (
            <div className="diagnosis-card" key={disease.id}>
              <div className="diagnosis-header">
                <div className="diagnosis-top">
                  <div>
                    <span className="rank-badge">
                      {idioma === "pt"
                        ? `#${index + 1} Diagnóstico`
                        : `#${index + 1} Diagnosis`}
                    </span>

                    <h2 className="disease-name">
                      {idioma === "pt"
                        ? disease.doencaPt
                        : disease.doencaEn}
                    </h2>

                    <div className="disease-en">
                      {idioma === "pt"
                        ? disease.doencaEn
                        : disease.doencaPt}
                    </div>
                  </div>

                  <div className="score">
                    <div className="score-value">
                      {(disease.score * 100).toFixed(2)}%
                    </div>

                    <small>
                      {idioma === "pt"
                        ? "Compatibilidade"
                        : "Compatibility"}
                    </small>
                  </div>
                </div>

                <div className="progress">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${Math.min(
                        disease.score * 100,
                        100
                      )}%`
                    }}
                  />
                </div>
              </div>

              <div className="diagnosis-content">
                <div className="section">
                  <h3>
                    {idioma === "pt" ? "Descrição" : "Description"}
                  </h3>

                  <p>
                    {idioma === "pt"
                      ? disease.descricaoPt
                      : disease.descricaoEn}
                  </p>
                </div>

                <div className="section">
                  <h3>
                    {idioma === "pt"
                      ? "Sintomas"
                      : "Symptoms"}
                  </h3>

                  <div className="symptoms">
                    {disease.symptoms.map(symptom => (
                      <span
                        className="symptom"
                        key={symptom}
                      >
                        {idioma === "pt"
                          ? symptomsMap[symptom]?.pt
                          : symptomsMap[symptom]?.en}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="section">
                  <h3>
                    {idioma === "pt"
                      ? "Tratamento"
                      : "Treatment"}
                  </h3>

                  <p>
                    {idioma === "pt"
                      ? disease.tratamentoPt
                      : disease.tratamentoEn}
                  </p>
                </div>

                {disease.highlyAssociatedWithSmoking && (
                  <div className="smoking-alert">
                    <h4>
                      {idioma === "pt"
                        ? "🚬 Associada ao Tabagismo"
                        : "🚬 Smoking Related"}
                    </h4>

                    <p>
                      {idioma === "pt"
                        ? "Esta doença possui forte relação com o hábito de fumar."
                        : "This disease has a strong association with smoking."}
                    </p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )
      }
    </div>
  );
}


export default App;
from enum import Enum

class PrivilegioEnum(str, Enum):
    admin     = "admin"
    usuario   = "usuario"
    avaliador = "avaliador"


class TituloEnum(str, Enum):
    MEST      = "MEST"       # Mestrando(a) / Aluno(a) de Mestrado
    DOUT      = "DOUT"       # Doutorando(a) / Aluno(a) de Doutorado
    PDOC      = "PDOC"       # Pesquisador(a) de Pós-Doutorado
    PROF_EF   = "PROF_EF"    # Professor(a) Efetivo
    PROF_VIS  = "PROF_VIS"   # Professor(a) Visitante
    PROF_SUB  = "PROF_SUB"   # Professor(a) Substituto(a)
    PG_MEST   = "PG_MEST"    # Aluno(a) de Pós-Graduação Mestrado
    PG_DOUT   = "PG_DOUT"    # Aluno(a) de Pós-Graduação Doutorado
    PESQ_VOL  = "PESQ_VOL"   # Professor/Pesquisador(a) Voluntário(a)
    ESPEC     = "ESPEC"      # Aluno(a) de Especialização
    EMER      = "EMER"       # Professor/Pesquisador(a) Emérito
    OUTRO     = "OUTRO"      # Outra


class DepartamentoEnum(str, Enum):
    # ── Campus / Faculdades ───────────────────────────────────────────────────
    FCTE          = "FCTE"           # Faculdade de Ciências e Tecnologias em Engenharia (Gama)
    FGA           = "FGA"            # Faculdade do Gama

    FACE          = "FACE"           # Faculdade de Economia, Administração e Contabilidade
    FACE_GPP      = "FACE-GPP"       # Departamento de Gestão de Políticas Públicas
    FACE_ADM      = "FACE-ADM"       # Departamento de Administração
    FACE_CCA      = "FACE-CCA"       # Departamento de Ciências Contábeis e Atuariais
    FACE_ECO      = "FACE-ECO"       # Departamento de Economia
    FACE_CIORD    = "FACE-CIORD"     # Centro Integrado de Ordenamento Territorial
    FACE_CEIGES   = "FACE-CEIGES"    # Centro de Estudos em Gestão em Saúde

    FCTS          = "FCTS"           # Faculdade de Ciências e Tecnologias em Saúde (Ceilândia)
    FCE           = "FCE"            # Faculdade de Ceilândia

    FAU           = "FAU"            # Faculdade de Arquitetura e Urbanismo
    FAU_PRO       = "FAU-PRO"        # Departamento de Projeto, Expressão e Representação
    FAU_TEC       = "FAU-TEC"        # Departamento de Tecnologia em Arquitetura e Urbanismo
    FAU_THAU      = "FAU-THAU"       # Departamento de Teoria e História em Arquitetura e Urbanismo
    FAU_CPAB      = "FAU-CPAB"       # Centro de Pesquisa e Aplicação de Bambu e Fibras Naturais

    FAV           = "FAV"            # Faculdade de Agronomia e Medicina Veterinária

    FAC           = "FAC"            # Faculdade de Comunicação
    FAC_DAP       = "FAC-DAP"        # Departamento de Audiovisuais e Publicidade
    FAC_JOR       = "FAC-JOR"        # Departamento de Jornalismo
    FAC_COM       = "FAC-COM"        # Departamento de Comunicação Organizacional

    FCI           = "FCI"            # Faculdade de Ciência da Informação
    FD            = "FD"             # Faculdade de Direito

    FE            = "FE"             # Faculdade de Educação
    FE_MTC        = "FE-MTC"         # Departamento de Métodos e Técnicas
    FE_PGE        = "FE-PGE"         # Departamento de Políticas Públicas e Gestão da Educação
    FE_TEF        = "FE-TEF"         # Departamento de Teoria e Fundamentos

    FEF           = "FEF"            # Faculdade de Educação Física
    FEF_CO        = "FEF-CO"         # Centro Olímpico

    FUP           = "FUP"            # Faculdade de Planaltina
    FM            = "FM"             # Faculdade de Medicina

    FS            = "FS"             # Faculdade de Ciências da Saúde
    FS_FAR        = "FS-FAR"         # Departamento de Farmácia
    FS_ODT        = "FS-ODT"         # Departamento de Odontologia
    FS_DSC        = "FS-DSC"         # Departamento de Saúde Coletiva
    FS_ENF        = "FS-ENF"         # Departamento de Enfermagem
    FS_NUT        = "FS-NUT"         # Departamento de Nutrição

    FT            = "FT"             # Faculdade de Tecnologia
    FT_EPR        = "FT-EPR"         # Departamento de Engenharia de Produção
    FT_EFL        = "FT-EFL"         # Departamento de Engenharia Florestal
    FT_ENC        = "FT-ENC"         # Departamento de Engenharia Civil e Ambiental
    FT_ENE        = "FT-ENE"         # Departamento de Engenharia Elétrica
    FT_ENM        = "FT-ENM"         # Departamento de Engenharia Mecânica
    FT_CEFTRU     = "FT-CEFTRU"      # Centro de Formação em Transportes

    # ── Institutos ────────────────────────────────────────────────────────────
    IB            = "IB"             # Instituto de Ciências Biológicas
    IB_BOT        = "IB-BOT"         # Departamento de Botânica
    IB_CEL        = "IB-CEL"         # Departamento de Biologia Celular
    IB_CFS        = "IB-CFS"         # Departamento de Ciências Fisiológicas
    IB_ECL        = "IB-ECL"         # Departamento de Ecologia
    IB_FIT        = "IB-FIT"         # Departamento de Fitopatologia
    IB_GEM        = "IB-GEM"         # Departamento de Genética e Morfologia
    IB_ZOO        = "IB-ZOO"         # Departamento de Zoologia
    IB_CNANO      = "IB-CNANO"       # Centro de Nano Ciência e Nanobiotecnologia
    IB_CP         = "IB-CP"          # Centro de Primatologia

    ICS           = "ICS"            # Instituto de Ciências Sociais
    ICS_ELA       = "ICS-ELA"        # Departamento de Estudos Latino-Americanos
    ICS_DAN       = "ICS-DAN"        # Departamento de Antropologia
    ICS_SOL       = "ICS-SOL"        # Departamento de Sociologia

    IDA           = "IdA"            # Instituto de Artes
    IDA_CEN       = "IdA-CEN"        # Departamento de Artes Cênicas
    IDA_DIN       = "IdA-DIN"        # Departamento de Desenho Industrial
    IDA_MUS       = "IdA-MUS"        # Departamento de Música
    IDA_VIS       = "IdA-VIS"        # Departamento de Artes Visuais

    IE            = "IE"             # Instituto de Ciências Exatas
    IE_CIC        = "IE-CIC"         # Departamento de Ciência da Computação
    IE_EST        = "IE-EST"         # Departamento de Estatística
    IE_MAT        = "IE-MAT"         # Departamento de Matemática

    IF            = "IF"             # Instituto de Física
    IF_CIF        = "IF-CIF"         # Centro Internacional de Física

    IG            = "IG"             # Instituto de Geociências
    IG_SIS        = "IG-SIS"         # Observatório Sismológico

    ICH           = "ICH"            # Instituto de Ciências Humanas
    ICH_FIL       = "ICH-FIL"        # Departamento de Filosofia
    ICH_GEA       = "ICH-GEA"        # Departamento de Geografia
    ICH_HIS       = "ICH-HIS"        # Departamento de História
    ICH_SER       = "ICH-SER"        # Departamento de Serviço Social

    IL            = "IL"             # Instituto de Letras
    IL_LET        = "IL-LET"         # Departamento de Línguas Estrangeiras e Tradução
    IL_LIP        = "IL-LIP"         # Departamento de Linguística, Português e Línguas Clássicas
    IL_TEL        = "IL-TEL"         # Departamento de Teoria Literária e Literatura

    IP            = "IP"             # Instituto de Psicologia
    IP_PST        = "IP-PST"         # Departamento de Psicologia Social e do Trabalho
    IP_PCL        = "IP-PCL"         # Departamento de Psicologia Clínica
    IP_PED        = "IP-PED"         # Departamento de Psicologia Escolar e do Desenvolvimento
    IP_PPB        = "IP-PPB"         # Departamento de Processos Psicológicos Básicos
    IP_CAEP       = "IP-CAEP"        # Centro de Atendimento e Estudos Psicológicos

    IPOL          = "IPOL"           # Instituto de Ciência Política
    IQ            = "IQ"             # Instituto de Química
    IREL          = "IREL"           # Instituto de Relações Internacionais


# ─── Project ──────────────────────────────────────────────────────────────────

class ProjectStatusEnum(str, Enum):
    rascunho   = "rascunho"
    em_analise = "em_analise"
    submetido  = "submetido"
    aprovado   = "aprovado"


class AreaCnpqEnum(str, Enum):
    EXATAS          = "EXATAS"           # Ciências Exatas e da Terra
    BIOLOGICAS      = "BIOLOGICAS"       # Ciências Biológicas
    ENGENHARIAS     = "ENGENHARIAS"      # Engenharias
    SAUDE           = "SAUDE"            # Ciências da Saúde
    AGRARIAS        = "AGRARIAS"         # Ciências Agrárias
    SOCIAIS_APLIC   = "SOCIAIS_APLIC"    # Ciências Sociais Aplicadas
    HUMANAS         = "HUMANAS"          # Ciências Humanas
    OUTRA           = "OUTRA"            # Outra


class SubAreaCnpqEnum(str, Enum):
    # Ciências Exatas e da Terra
    MAT        = "MAT"        # Matemática
    PROB_EST   = "PROB_EST"   # Probabilidade e Estatística
    COMP       = "COMP"       # Ciência da Computação
    ASTRO      = "ASTRO"      # Astronomia
    FIS        = "FIS"        # Física
    QUIM       = "QUIM"       # Química
    GEO        = "GEO"        # Geociências
    OCEAN      = "OCEAN"      # Oceanografia

    # Ciências Biológicas
    BIO_GERAL  = "BIO_GERAL"  # Biologia Geral
    GENET      = "GENET"      # Genética
    BOT_B      = "BOT_B"      # Botânica
    ZOO_B      = "ZOO_B"      # Zoologia
    ECOL       = "ECOL"       # Ecologia
    MORF       = "MORF"       # Morfologia
    FISIOL     = "FISIOL"     # Fisiologia
    BIOQUIM    = "BIOQUIM"    # Bioquímica
    BIOFIS     = "BIOFIS"     # Biofísica
    FARMACOL   = "FARMACOL"   # Farmacologia
    IMUNOL     = "IMUNOL"     # Imunologia
    MICROBIOL  = "MICROBIOL"  # Microbiologia
    PARASITOL  = "PARASITOL"  # Parasitologia
    BIOL_MOL   = "BIOL_MOL"   # Biologia Molecular

    # Engenharias
    ENG_CIV    = "ENG_CIV"    # Engenharia Civil
    ENG_MIN    = "ENG_MIN"    # Engenharia de Minas
    ENG_MAT    = "ENG_MAT"    # Engenharia de Materiais e Metalúrgica
    ENG_ELE    = "ENG_ELE"    # Engenharia Elétrica
    ENG_MEC    = "ENG_MEC"    # Engenharia Mecânica
    ENG_QUIM   = "ENG_QUIM"   # Engenharia Química
    ENG_SAN    = "ENG_SAN"    # Engenharia Sanitária
    ENG_PROD   = "ENG_PROD"   # Engenharia de Produção
    ENG_NUC    = "ENG_NUC"    # Engenharia Nuclear
    ENG_TRANSP = "ENG_TRANSP" # Engenharia de Transportes
    ENG_NAV    = "ENG_NAV"    # Engenharia Naval e Oceânica
    ENG_AERO   = "ENG_AERO"   # Engenharia Aeroespacial
    ENG_BIOMED = "ENG_BIOMED" # Engenharia Biomédica

    # Ciências da Saúde
    MED        = "MED"        # Medicina
    ODONT      = "ODONT"      # Odontologia
    FARM       = "FARM"       # Farmácia
    ENFERM     = "ENFERM"     # Enfermagem
    NUTRI      = "NUTRI"      # Nutrição
    SAUDE_COL  = "SAUDE_COL"  # Saúde Coletiva
    FONOA      = "FONOA"      # Fonoaudiologia
    FISIOTER   = "FISIOTER"   # Fisioterapia e Terapia Ocupacional
    ED_FIS     = "ED_FIS"     # Educação Física

    # Ciências Agrárias
    AGRON      = "AGRON"      # Agronomia
    REC_FLOR   = "REC_FLOR"   # Recursos Florestais e Engenharia Florestal
    ENG_AGRIC  = "ENG_AGRIC"  # Engenharia Agrícola
    ZOOTEC     = "ZOOTEC"     # Zootecnia
    MED_VET    = "MED_VET"    # Medicina Veterinária
    REC_PESC   = "REC_PESC"   # Recursos Pesqueiros e Engenharia de Pesca
    CI_ALIM    = "CI_ALIM"    # Ciência e Tecnologia de Alimentos

    # Ciências Sociais Aplicadas
    DIR        = "DIR"        # Direito
    ADM_S      = "ADM_S"      # Administração
    ECON_S     = "ECON_S"     # Economia
    ARQ_URB    = "ARQ_URB"    # Arquitetura e Urbanismo
    PLAN_URB   = "PLAN_URB"   # Planejamento Urbano e Regional
    COMUN      = "COMUN"      # Comunicação
    CI_INF     = "CI_INF"     # Ciência da Informação
    MUSEOL     = "MUSEOL"     # Museologia
    SERV_SOC   = "SERV_SOC"   # Serviço Social
    ECON_DOM   = "ECON_DOM"   # Economia Doméstica
    DES_IND    = "DES_IND"    # Desenho Industrial
    TURISMO    = "TURISMO"    # Turismo

    # Ciências Humanas
    FILOS      = "FILOS"      # Filosofia
    SOCIOL     = "SOCIOL"     # Sociologia
    ANTROP     = "ANTROP"     # Antropologia
    ARQUEOL    = "ARQUEOL"    # Arqueologia
    HIST       = "HIST"       # História
    GEOGR      = "GEOGR"      # Geografia
    PSICOL     = "PSICOL"     # Psicologia
    EDUC       = "EDUC"       # Educação
    CI_POL     = "CI_POL"     # Ciência Política
    TEOL       = "TEOL"       # Teologia
    LING       = "LING"       # Linguística
    LIT        = "LIT"        # Literatura
    ARTES      = "ARTES"      # Artes

    # Outra
    OUTRA_SUB  = "OUTRA_SUB"  # Outra


SUBAREA_TO_AREA: dict[SubAreaCnpqEnum, AreaCnpqEnum] = {
    # Exatas
    SubAreaCnpqEnum.MAT:        AreaCnpqEnum.EXATAS,
    SubAreaCnpqEnum.PROB_EST:   AreaCnpqEnum.EXATAS,
    SubAreaCnpqEnum.COMP:       AreaCnpqEnum.EXATAS,
    SubAreaCnpqEnum.ASTRO:      AreaCnpqEnum.EXATAS,
    SubAreaCnpqEnum.FIS:        AreaCnpqEnum.EXATAS,
    SubAreaCnpqEnum.QUIM:       AreaCnpqEnum.EXATAS,
    SubAreaCnpqEnum.GEO:        AreaCnpqEnum.EXATAS,
    SubAreaCnpqEnum.OCEAN:      AreaCnpqEnum.EXATAS,
    # Biológicas
    SubAreaCnpqEnum.BIO_GERAL:  AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.GENET:      AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.BOT_B:      AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.ZOO_B:      AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.ECOL:       AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.MORF:       AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.FISIOL:     AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.BIOQUIM:    AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.BIOFIS:     AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.FARMACOL:   AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.IMUNOL:     AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.MICROBIOL:  AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.PARASITOL:  AreaCnpqEnum.BIOLOGICAS,
    SubAreaCnpqEnum.BIOL_MOL:   AreaCnpqEnum.BIOLOGICAS,
    # Engenharias
    SubAreaCnpqEnum.ENG_CIV:    AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_MIN:    AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_MAT:    AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_ELE:    AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_MEC:    AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_QUIM:   AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_SAN:    AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_PROD:   AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_NUC:    AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_TRANSP: AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_NAV:    AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_AERO:   AreaCnpqEnum.ENGENHARIAS,
    SubAreaCnpqEnum.ENG_BIOMED: AreaCnpqEnum.ENGENHARIAS,
    # Saúde
    SubAreaCnpqEnum.MED:        AreaCnpqEnum.SAUDE,
    SubAreaCnpqEnum.ODONT:      AreaCnpqEnum.SAUDE,
    SubAreaCnpqEnum.FARM:       AreaCnpqEnum.SAUDE,
    SubAreaCnpqEnum.ENFERM:     AreaCnpqEnum.SAUDE,
    SubAreaCnpqEnum.NUTRI:      AreaCnpqEnum.SAUDE,
    SubAreaCnpqEnum.SAUDE_COL:  AreaCnpqEnum.SAUDE,
    SubAreaCnpqEnum.FONOA:      AreaCnpqEnum.SAUDE,
    SubAreaCnpqEnum.FISIOTER:   AreaCnpqEnum.SAUDE,
    SubAreaCnpqEnum.ED_FIS:     AreaCnpqEnum.SAUDE,
    # Agrárias
    SubAreaCnpqEnum.AGRON:      AreaCnpqEnum.AGRARIAS,
    SubAreaCnpqEnum.REC_FLOR:   AreaCnpqEnum.AGRARIAS,
    SubAreaCnpqEnum.ENG_AGRIC:  AreaCnpqEnum.AGRARIAS,
    SubAreaCnpqEnum.ZOOTEC:     AreaCnpqEnum.AGRARIAS,
    SubAreaCnpqEnum.MED_VET:    AreaCnpqEnum.AGRARIAS,
    SubAreaCnpqEnum.REC_PESC:   AreaCnpqEnum.AGRARIAS,
    SubAreaCnpqEnum.CI_ALIM:    AreaCnpqEnum.AGRARIAS,
    # Sociais Aplicadas
    SubAreaCnpqEnum.DIR:        AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.ADM_S:      AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.ECON_S:     AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.ARQ_URB:    AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.PLAN_URB:   AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.COMUN:      AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.CI_INF:     AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.MUSEOL:     AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.SERV_SOC:   AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.ECON_DOM:   AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.DES_IND:    AreaCnpqEnum.SOCIAIS_APLIC,
    SubAreaCnpqEnum.TURISMO:    AreaCnpqEnum.SOCIAIS_APLIC,
    # Humanas
    SubAreaCnpqEnum.FILOS:      AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.SOCIOL:     AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.ANTROP:     AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.ARQUEOL:    AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.HIST:       AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.GEOGR:      AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.PSICOL:     AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.EDUC:       AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.CI_POL:     AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.TEOL:       AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.LING:       AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.LIT:        AreaCnpqEnum.HUMANAS,
    SubAreaCnpqEnum.ARTES:      AreaCnpqEnum.HUMANAS,
    # Outra
    SubAreaCnpqEnum.OUTRA_SUB:  AreaCnpqEnum.OUTRA,
}

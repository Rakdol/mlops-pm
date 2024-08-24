import enum


class PLATFORM_ENUM(enum.Enum):
    DOCKER = "docker"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    TEST = "test"

    @staticmethod
    def has_value(item):
        return item in [v.value for v in PLATFORM_ENUM.__members__.values()]


class MODEL_ENUM(enum.Enum):
    LOGIT_MODEL = "logit"
    RF_MODEL = "rf"

    @staticmethod
    def has_value(item):
        return item in [v.value for v in MODEL_ENUM.__members__.values()]


class CV_ENUM(enum.Enum):
    simple_CV = "cv"
    strat_cv = "strat_cv"

    @staticmethod
    def has_value(item):
        return item in [v.value for v in CV_ENUM.__members__.values()]

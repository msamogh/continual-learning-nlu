from typing import *

from cl_domain.domain import Domain
from cl_domain.utils import GLOBAL_RAND


def random_ordering(domains: Dict[Text, Domain]) -> List[Domain]:
    """Generate a random ordering of domains."""
    domains = list(domains.values())
    GLOBAL_RAND.shuffle(domains)
    return domains


def max_path_ordering(domains: List[Domain]) -> List[Domain]:
    """Generate a domain ordering that maximizes the number of paths between
    domains.
    """
    raise NotImplementedError


def min_path_ordering(domains: List[Domain]) -> List[Domain]:
    """Generate a domain ordering that minimizes the number of paths between
    domains.
    """
    raise NotImplementedError

"""Who is logged in? Turns a login token into a member_id.

SECURITY (the core rule of this project): member_id comes ONLY from here, from
the verified login token. Never from the URL, the request body or the chat
text. So no prompt, and no edited request, can make the API act as someone else.

CONCEPT: in production the token would be a signed JWT from a real identity
provider (Cognito, Auth0, Okta), and we'd verify its signature and expiry. For
the MVP a fake lookup table stands in, but the SHAPE of the code is the same:
the rest of the app just asks for "the current member" and never cares how
the token was checked. Swapping in real auth later changes only this file.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# token -> member_id. Stand-in for a real identity provider.
FAKE_TOKENS = {
    "token-asha": "M123",
    "token-ben": "M456",
}

# CONCEPT (HTTP): clients send   Authorization: Bearer <token>   in the request
# headers. HTTPBearer reads and parses that header for us, AND makes Swagger UI
# (/docs) show an "Authorize" button where you can paste a token.
# auto_error=False: if the header is missing, give us None instead of failing,
# so we can return our own consistent 401 below.
bearer_scheme = HTTPBearer(auto_error=False)


# CONCEPT (FastAPI): a DEPENDENCY is a function FastAPI runs BEFORE your endpoint,
# passing its return value in as an argument. Any endpoint that declares
# `member_id: CurrentMember` automatically gets authentication.
# Dependencies can depend on other dependencies: this one depends on bearer_scheme.
#
# PYTHON: Annotated[Type, extra-info] attaches metadata to a type hint. Here the
# type is HTTPAuthorizationCredentials | None, and the metadata Depends(...) tells
# FastAPI where the value comes from.
async def get_current_member_id(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    if creds is None or creds.credentials not in FAKE_TOKENS:
        # CONCEPT (HTTP): 401 Unauthorized = "I don't know who you are" (no or bad
        # credentials). 403 Forbidden = "I know who you are, but you can't do this".
        # The WWW-Authenticate header tells the client which auth scheme to use.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid login token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return FAKE_TOKENS[creds.credentials]


# PYTHON: a TYPE ALIAS. Writing `member_id: CurrentMember` in an endpoint is short
# for `member_id: Annotated[str, Depends(get_current_member_id)]`.
CurrentMember = Annotated[str, Depends(get_current_member_id)]

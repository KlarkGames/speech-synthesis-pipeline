"""feat: added CPS metric for ASR and Original texts

Revision ID: dead6ba9eb5f
Revises: 
Create Date: 2025-03-10 13:33:40.267293

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dead6ba9eb5f'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('audio_to_original_text', sa.Column('cps', sa.Float()))
    op.add_column('audio_to_asr_text', sa.Column('cps', sa.Float()))


def downgrade() -> None:
    op.drop_column('audio_to_original_text', 'cps')
    op.drop_column('audio_to_asr_text', 'cps')

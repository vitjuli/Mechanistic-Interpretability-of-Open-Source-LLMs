import { BEHAVIOUR_TYPE_COLORS, BEHAVIOUR_TYPE_LABELS } from '../../utils/colors';

/**
 * Small pill badge showing the mechanistic behaviour type.
 * Renders nothing if behaviourType is null/undefined.
 */
export default function BehaviourTypePill({ behaviourType, style = {} }) {
  if (!behaviourType) return null;
  const color = BEHAVIOUR_TYPE_COLORS[behaviourType] ?? '#888';
  const label = BEHAVIOUR_TYPE_LABELS[behaviourType] ?? behaviourType;
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      fontSize: 10,
      fontWeight: 600,
      padding: '2px 7px',
      borderRadius: 10,
      background: `${color}22`,
      border: `1px solid ${color}66`,
      color,
      letterSpacing: '0.02em',
      whiteSpace: 'nowrap',
      ...style,
    }}>
      {label}
    </span>
  );
}

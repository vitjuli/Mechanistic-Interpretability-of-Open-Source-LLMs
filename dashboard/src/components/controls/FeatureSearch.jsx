import useStore from '../../store/useStore';

export default function FeatureSearch() {
  const searchQuery = useStore(s => s.searchQuery);
  const setSearchQuery = useStore(s => s.setSearchQuery);

  return (
    <div className="control-group">
      <label>Search Features</label>
      <input
        type="text"
        placeholder="e.g. L15_F124340 or 124340"
        value={searchQuery}
        onChange={e => setSearchQuery(e.target.value)}
      />
    </div>
  );
}

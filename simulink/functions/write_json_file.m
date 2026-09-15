function write_json_file(value,path)
%WRITE_JSON_FILE Write pretty JSON using MATLAB's native encoder.
folder=fileparts(path);
if ~isempty(folder) && ~exist(folder,'dir'), mkdir(folder); end
text=jsonencode(value,'PrettyPrint',true);
fid=fopen(path,'w');
if fid<0, error('retina_nexus:JsonWrite','Cannot write %s',path); end
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',text);
end

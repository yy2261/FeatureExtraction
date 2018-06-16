package getJavaFile;

import java.io.File;
import java.util.LinkedList;

import DemoVisitorTest.DemoVisitorTest;

public class GetJavaFile {
	
	public static void getFile(File file, LinkedList<String> pathList){
		File fileList[] = file.listFiles();
		for(int i = 0; i<fileList.length; i++){
			if(fileList[i].isFile()){
				if(fileList[i].toString().indexOf(".java") != -1)
					pathList.add(fileList[i].toString());
			}
			if(fileList[i].isDirectory()){
				getFile(fileList[i], pathList);
			}
		}
	}
	
	public static void main(String[] args) throws Exception{
		String dir = "/home/yy/apache-ant-1.10.1/";
		File file = new File(dir);
		LinkedList<String> pathList = new LinkedList<String> ();
		getFile(file, pathList);
		for(int i = 0; i < pathList.size(); i++)
		{
			String filename = pathList.get(i).substring(1, pathList.get(i).length());
			String s = "/home/yy/features/"+filename.replace("/", ".");
			DemoVisitorTest.parse(pathList.get(i), s);
	//		System.out.println(pathList.get(i));
		}
	}

}

package animacy;

import java.io.File;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.ISynsetID;
import edu.mit.jwi.item.IWord;
import edu.mit.jwi.item.IWordID;
import edu.mit.jwi.item.POS;
import edu.mit.jwi.item.Pointer;

/**
 The GetWordnetFeatures class parse hypernym feature from Wordnet for hybrid model 
 **/

public class GetWordnetFeatures 
{
	static String projectPath = System.getProperty("user.dir");
    static File projectDir = new File(projectPath);
    static File wordnetProj = new File(projectDir, "lib/WordNet"); // had to remove projectDir.getParent()
    static File dictDir = new File(wordnetProj, "/3.0/dict");
    

	//get wordnet feature 
	public static int getAll(String text, IDictionary dict)
	{

		// get the synset
		 IIndexWord idxWord = dict.getIndexWord (text, POS.NOUN);
		 if(idxWord == null)	 return 0;
		 IWordID wordID = idxWord.getWordIDs().get(0) ; // 1st meaning
		 IWord word = dict.getWord (wordID);
		 ISynset synset = word.getSynset();
		
		 // get the hypernyms
		ISynsetID hypernymID = null;
		if(!synset.getRelatedSynsets(Pointer.HYPERNYM ).isEmpty()){
	        hypernymID =  synset.getRelatedSynsets(Pointer.HYPERNYM ).get(0);
	 		IWord word1 = dict.getSynset(hypernymID).getWords().get(0);
	 		Boolean animate = false, entity = false;
	 		
	 		while(word1.getSynset() != null)
	 		{
	 			synset = word1.getSynset();
	 			hypernymID = synset.getRelatedSynsets(Pointer.HYPERNYM).get(0);
	 			word1 = dict.getSynset(hypernymID).getWords().get(0);
	 			
	 			//if a word belongs to living thing, mark it as animate, otherwise mark as inanimate
	 			if(word1.getLemma().equals("living_thing")){
	 				animate = true;
	 				break;
	 			}if(word1.getLemma().equals("entity")){
	 				entity = true;
	 				break;
	 			}
	 		}
	 		
	 		if(animate == true)		return 1;
	 		else if(entity==true)	return 2;
	 		else	return 0;
		}
		else{
			return 0;
		}
					
	}
		   
}

		 


	




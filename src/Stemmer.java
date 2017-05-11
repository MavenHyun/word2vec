import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class Stemmer {
    private StringBuilder buffer;
    private int offset, /* offset into buffer */
            offset_end, /* offset to end of stemmed word */
            current, // Current letter
            end; // End of stem

    public Stemmer() {
        buffer = new StringBuilder();
        offset = 0;
        offset_end = 0;
    }

    public void clear() {
        buffer = new StringBuilder();
        offset = 0;
        offset_end = 0;
    }

    /**
     * @param ch
     *            Character to add Add a character to the word being stemmed.
     *            When you are finished adding characters, you can call
     *            stem(void) to stem the word.
     */

    public void add(char ch) {
        buffer.append(ch);
        offset++;
    }

    /**
     * @param chars
     *            Add many characters to the buffer at once.
     */
    public void add(char[] chars) {
        buffer.append(chars);
        offset += chars.length;
    }

    /**
     * @param seq
     *            CharSequence to be added to the buffer.
     */
    public void add(CharSequence seq) {
        buffer.append(seq);
        offset += seq.length();
    }

    /**
     * After a word has been stemmed, it can be retrieved by toString(), or a
     * reference to the internal buffer can be retrieved by getResultBuffer and
     * getResultLength (which is generally more efficient.)
     *
     * Also, stem() returns the stemmed string.
     */
    public String toString() {
        return buffer.substring(0, end + 1);
    }

    /**
     * @return the length of the word resulting from the stemming process.
     */
    public int getResultLength() {
        return offset_end;
    }

    /**
     * @return a reference to a character buffer containing the results of the
     *         stemming process.
     */
    public char[] getResultBuffer() {
        return buffer.toString().toCharArray();
    }

    /**
     *
     * @param i
     *            index of the character to check.
     * @return true if character is a consonant, false otherwise.
     */
    private final boolean isConsonant(int i) {

        switch (Character.toLowerCase(buffer.charAt(i))) {
            case 'a':
            case 'e':
            case 'i':
            case 'o':
            case 'u':
                return false;
            case 'y':
                return (i == 0) ? true : !isConsonant(i - 1);
            default:
                return true;
        }
    }

    /**
     * @return the number of consonant sequences between 0 and j. if c is a
     *         consonant sequence and v a vowel sequence, and <..> indicates
     *         arbitrary presence,
     *
     *         <c><v> gives 0 <c>vc<v> gives 1 <c>vcvc<v> gives 2 <c>vcvcvc<v>
     *         gives 3 ....
     */
    private final int getNumberOfConsonantSequences() {
        int numSequences = 0;
        int index = 0;
        while (isConsonant(index)) {
            index++;
            if (index > current)
                return numSequences;
        }
        index++;
        while (index <= current) {
            while (!isConsonant(index)) {
                index++;
                if (index > current)
                    return numSequences;
            }
            index++;
            numSequences++;
            while (isConsonant(index)) {
                index++;
                if (index > current)
                    return numSequences;
            }
            index++;
        }
        return numSequences;
    }

    /**
     * @return true <=> 0,...j contains a vowel
     *
     */

    private final boolean hasVowelInStem() {
        for (int i = 0; i <= current; i++) {
            if (!isConsonant(i)) {
                return true;
            }
        }
        return false;
    }

    /**
     * @param index
     *            check buffer at index and index-1 to see if they are both
     *            consonants
     * @return true <=> index,(index-1) contain a double consonant.
     */

    private final boolean isDoubleConsonant(int index) {
        if (index < 1 || buffer.charAt(index) != buffer.charAt(index - 1))
            return false;

        return isConsonant(index);
    }

    /**
     * @param i
     *            last of the three characters to be checked to see if they are
     *            a consonant - vowel - consonant sequence.
     *
     * @return true <=> i-2,i-1,i has the form consonant - vowel - consonant and
     *         also if the second consonant is not w,x or y. This is used when
     *         trying to restore an e at the end of a short word. e.g.
     *
     *         cav(e), lov(e), hop(e), crim(e), but not snow, box, tray.
     */

    private final boolean isCVCSequence(int i) {
        if (i < 2 || !isConsonant(i) || isConsonant(i - 1)
                || !isConsonant(i - 2))
            return false;

        char ch = Character.toLowerCase(buffer.charAt(i));
        if (ch == 'w' || ch == 'x' || ch == 'y')
            return false;

        return true;
    }

    /**
     *
     * @param s
     *            ending to compare to the buffer's ending.
     * @return true if the buffer ends in s
     */
    private final boolean endsIn(String s) {
        s = s.toLowerCase();
        int len = s.length();
        int o = end - len + 1;
        if (o < 0)
            return false;
        for (int i = 0; i < len; i++)
            if (Character.toLowerCase(buffer.charAt(o + i)) != s.charAt(i)) {
                return false;
            }
        current = end - len;
        return true;
    }

    /**
     * @param s
     *            string to set current+1 .. ending to.
     */

    private final void setTo(String s) {
        int len = s.length();
        int o = current + 1;
        for (int i = 0; i < len; i++)
            buffer.setCharAt(o + i, s.charAt(i));
        end = current + len;
    }

    /**
     * @param s
     *            setTo(s) if and only if there are any consonant sequences
     */
    private final void replaceWith(String s) {
        if (getNumberOfConsonantSequences() > 0)
            setTo(s);
    }

    /**
     * gets rid of plurals and -ed or -ing. e.g.
     *
     * caresses -> caress, ponies -> poni, ties -> ti, caress -> caress, cats ->
     * cat,
     *
     * feed -> feed, agreed -> agree, disabled -> disable,
     *
     * matting -> mat, mating -> mate, meeting -> meet, milling -> mill, messing
     * -> mess,
     *
     * meetings -> meet
     */

    private final void removePluralOrEdOrIng() {
        char c = Character.toLowerCase(buffer.charAt(end));
        if (c == 's') {
            if (endsIn("sses"))
                end -= 2;
            else if (endsIn("ies"))
                setTo("i");
            else if (Character.toLowerCase(buffer.charAt(end - 1)) != 's')
                end--;
        }
        if (endsIn("eed")) {
            if (getNumberOfConsonantSequences() > 0)
                end--;
        } else if ((endsIn("ed") || endsIn("ing")) && hasVowelInStem()) {

            end = current;
            if (endsIn("at"))
                setTo("ate");
            else if (endsIn("bl"))
                setTo("ble");
            else if (endsIn("iz"))
                setTo("ize");
            else if (isDoubleConsonant(end)) {
                end--;
                {
                    int ch = Character.toLowerCase(buffer.charAt(end));
                    if (ch == 'l' || ch == 's' || ch == 'z')
                        end++;
                }
            } else if (getNumberOfConsonantSequences() == 1
                    && isCVCSequence(end))
                setTo("e");
        }
    }

    /**
     * turns terminal y to i when there is another vowel in the stem.
     */

    private final void convertTerminalY() {
        if (endsIn("y") && hasVowelInStem())
            buffer.setCharAt(end, 'i');

    }

    /**
     * maps double suffices to single ones. so -ization ( = -ize plus -ation)
     * maps to -ize etc. note that the string before the suffix must give
     * getNumberOfConsonantSequences() > 0.
     */

    private final void doubleToSingleSuffices() {
        if (end == 0)
            return; /* For Bug 1 */
        switch (Character.toLowerCase(buffer.charAt(end - 1))) {
            case 'a':
                if (endsIn("ational")) {
                    replaceWith("ate");
                    break;
                }
                if (endsIn("tional")) {
                    replaceWith("tion");
                    break;
                }
                break;
            case 'c':
                if (endsIn("enci")) {
                    replaceWith("ence");
                    break;
                }
                if (endsIn("anci")) {
                    replaceWith("ance");
                    break;
                }
                break;
            case 'e':
                if (endsIn("izer")) {
                    replaceWith("ize");
                    break;
                }
                break;
            case 'l':
                if (endsIn("bli")) {
                    replaceWith("ble");
                    break;
                }
                if (endsIn("alli")) {
                    replaceWith("al");
                    break;
                }
                if (endsIn("entli")) {
                    replaceWith("ent");
                    break;
                }
                if (endsIn("eli")) {
                    replaceWith("e");
                    break;
                }
                if (endsIn("ousli")) {
                    replaceWith("ous");
                    break;
                }
                break;
            case 'o':
                if (endsIn("ization")) {
                    replaceWith("ize");
                    break;
                }
                if (endsIn("ation")) {
                    replaceWith("ate");
                    break;
                }
                if (endsIn("ator")) {
                    replaceWith("ate");
                    break;
                }
                break;
            case 's':
                if (endsIn("alism")) {
                    replaceWith("al");
                    break;
                }
                if (endsIn("iveness")) {
                    replaceWith("ive");
                    break;
                }
                if (endsIn("fulness")) {
                    replaceWith("ful");
                    break;
                }
                if (endsIn("ousness")) {
                    replaceWith("ous");
                    break;
                }
                break;
            case 't':
                if (endsIn("aliti")) {
                    replaceWith("al");
                    break;
                }
                if (endsIn("iviti")) {
                    replaceWith("ive");
                    break;
                }
                if (endsIn("biliti")) {
                    replaceWith("ble");
                    break;
                }
                break;
            case 'g':
                if (endsIn("logi")) {
                    replaceWith("log");
                    break;
                }
        }
    }

    /**
     * step4() deals with -ic-, -full, -ness etc. similar strategy to step3.
     */

    private final void removeSuffices() {
        switch (Character.toLowerCase(buffer.charAt(end))) {
            case 'e':
                if (endsIn("icate")) {
                    replaceWith("ic");
                    break;
                }
                if (endsIn("ative")) {
                    replaceWith("");
                    break;
                }
                if (endsIn("alize")) {
                    replaceWith("al");
                    break;
                }
                break;
            case 'i':
                if (endsIn("iciti")) {
                    replaceWith("ic");
                    break;
                }
                break;
            case 'l':
                if (endsIn("ical")) {
                    replaceWith("ic");
                    break;
                }
                if (endsIn("ful")) {
                    replaceWith("");
                    break;
                }
                break;
            case 's':
                if (endsIn("ness")) {
                    replaceWith("");
                    break;
                }
                break;
        }
    }

    /**
     * takes off -ant, -ence etc., in context <c>vcvc<v>.
     */

    private final void removeVCVCEndings() {
        if (end == 0)
            return; /* for Bug 1 */
        switch (Character.toLowerCase(buffer.charAt(end - 1))) {
            case 'a':
                if (endsIn("al"))
                    break;
                return;
            case 'c':
                if (endsIn("ance"))
                    break;
                if (endsIn("ence"))
                    break;
                return;
            case 'e':
                if (endsIn("er"))
                    break;
                return;
            case 'i':
                if (endsIn("ic"))
                    break;
                return;
            case 'l':
                if (endsIn("able"))
                    break;
                if (endsIn("ible"))
                    break;
                return;
            case 'n':
                if (endsIn("ant"))
                    break;
                if (endsIn("ement"))
                    break;
                if (endsIn("ment"))
                    break;
			/* element etc. not stripped before the m */
                if (endsIn("ent"))
                    break;
                return;
            case 'o':
                if (endsIn("ion")
                        && current >= 0
                        && (Character.toLowerCase(buffer.charAt(current)) == 's' || Character
                        .toLowerCase(buffer.charAt(current)) == 't'))
                    break;
			/* j >= 0 fixes Bug 2 */
                if (endsIn("ou"))
                    break;
                return;
			/* takes care of -ous */
            case 's':
                if (endsIn("ism"))
                    break;
                return;
            case 't':
                if (endsIn("ate"))
                    break;
                if (endsIn("iti"))
                    break;
                return;
            case 'u':
                if (endsIn("ous"))
                    break;
                return;
            case 'v':
                if (endsIn("ive"))
                    break;
                return;
            case 'z':
                if (endsIn("ize"))
                    break;
                return;
            default:
                return;
        }
        if (getNumberOfConsonantSequences() > 1)
            end = current;
    }

    /**
     *  removes a final -e if number of consonant sequences > 1.
     */

    private final void removeFinalE() {
        current = end;
        char ch = Character.toLowerCase(buffer.charAt(end));
        if (ch == 'e') {
            int a = getNumberOfConsonantSequences();
            if (a > 1 || a == 1 && !isCVCSequence(end - 1))
                end--;
        }
        if (ch == 'l' && isDoubleConsonant(end)
                && getNumberOfConsonantSequences() > 1)
            end--;
    }

    /**
     * Stem the word placed into the Stemmer buffer through calls to add().
     * Returns true if the stemming process resulted in a word different from
     * the input. You can retrieve the result with
     * getResultLength()/getResultBuffer() or toString().
     */
    public String stem() {
        end = offset - 1;
        if (end > 1) {
            removePluralOrEdOrIng();
            convertTerminalY();
            doubleToSingleSuffices();
            removeSuffices();
            removeVCVCEndings();
            removeFinalE();
        }
        offset_end = end + 1;
        offset = 0;

        return buffer.substring(0, end + 1);
    }
}